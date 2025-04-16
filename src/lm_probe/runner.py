from dataclasses import dataclass
from collections import defaultdict
from typing import TYPE_CHECKING, Generator, Optional, Self

import torch
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from torch.utils.data import DataLoader

from .probe import LinearProbe, ProbeConfig
from .extractor import FeatureExtractor
from .dataset import ProbeDataset
from .metrics import Metrics
from .utils import setup_logger

if TYPE_CHECKING:
    from nnsight import LanguageModel


@dataclass
class BatchPredictions:
    """Probe predictions, probabilities, and labels for a single batch."""

    preds: NDArray[np.integer]
    probs: NDArray[np.floating]
    labels: NDArray[np.integer]


class ProbeRunner:
    """Runner for training multiple linear probes at once."""

    logger = setup_logger(__name__)

    def __init__(
        self,
        model: "LanguageModel",
        probe_configs: list[ProbeConfig],
        remote: bool = False,
    ) -> None:
        """Initialize the runner.

        Parameters
        ----------
        model : LanguageModel
            Langauge model
        prob_configs : list[ProbeConfig]
            Probe configurations
        remote : bool
            Whether to run the model remotely on NDIF (requires API key)
        """
        # Initialize the FeatureExtractor
        self.feature_extractor: FeatureExtractor = FeatureExtractor(
            model, remote=remote
        )

        # Set up the probes
        self.probes: list[LinearProbe] = [
            LinearProbe(config) for config in probe_configs
        ]
        self.submodules: list[str] = [probe.submodule for probe in self.probes]
        self._probe_map = {
            probe.submodule: idx for idx, probe in enumerate(self.probes)
        }

        # Are we pooling the features?
        pooling = set(config.pool for config in probe_configs)
        if len(pooling) > 1:
            raise ValueError(
                "Mixed values in probe pooling flags. All must match"
            )
        (self.pool,) = pooling

        # Set up metrics tracking
        self.metrics: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        """Repr method.

        Returns
        -------
        str
            Basic info about the runner
        """
        return f"Probe runner for {len(self.submodules)} probe(s)"

    def __getitem__(self, submodule: str) -> LinearProbe:
        """Get a probe.

        Parameters
        ----------
        submodule : str
            Name of the probe

        Returns
        -------
        LinearProbe
            The probe
        """
        return self.probes[self._probe_map[submodule]]

    def get_probe_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subset: list[str] = [],
    ) -> dict[str, NDArray[np.floating]]:
        """Get features from LanguageModel submodules that correspond to
        each probe.

        Parameters
        ----------
        input_ids : torch.Tensor
            Batch of token input IDs with shape (batch_size, seq_len, num_tok)
        attention_mask : torch.Tensor
            Attention mask for the features' tokens with shape (batch_size,
            seq_len)
        subset : list[str]
            Specific submodules, retrieves all if empty list

        Returns
        -------
        dict[str, np.ndarray]
            Output dictionary of submodule : feature pairs
        """
        submodules = subset if subset else self.submodules
        features = self.feature_extractor(
            submodules,
            input_ids,
            attention_mask=attention_mask,
            pool=self.pool,
            cache=False,
        )

        return {
            submodule: feat.detach().cpu().float().numpy()
            for submodule, feat in features.items()
        }

    def fit_probes(
        self,
        train_set: ProbeDataset,
        eval_set: ProbeDataset,
        num_epoch: int = 10,
        batch_size: int = 32,
        limit_eval_batches: bool = True,
    ) -> Self:
        """Fit the probes.

        Parameters
        ----------
        train_set : ProbeDataset
            Training dataset
        eval_set : ProbeDataset
            Evaluation dataset
        num_epoch : int
            Number of epochs to train
        batch_size : int
            Batch size for the model
        limit_eval_batches: bool
            Limit evaluation batches to the maximum number of batches seen
            while fitting
        """
        # For num_epochs...
        batch_count = 0
        for epoch in range(num_epoch):
            # Set up a DataLoader to batch out spans
            dataloader = DataLoader(train_set, batch_size=batch_size)

            # For every batch...
            for batch in dataloader:
                # Do we have trainable probes?
                trainable = [
                    probe for probe in self.probes if probe.is_trainable
                ]
                if not trainable:
                    break

                # Get probe features for the batch
                features = self.get_probe_features(
                    batch["input_ids"],
                    batch["attention_mask"],
                    [probe.submodule for probe in trainable],
                )
                labels = batch["labels"].numpy()

                # Take a step for every trainable probe
                for probe in trainable:
                    probe.take_step(features[probe.submodule], labels)

                batch_count += 1

            # End training before num_epochs is exceeded if there aren't any
            # trainable probes
            if not any(probe.is_trainable for probe in self.probes):
                break

        self.logger.info(
            "Finished training %d probe(s). Computing metrics",
            len(self.probes),
        )  # type: ignore[attr-defined]

        # Optionally limit the evaluation batches to the number of batches seen
        # during training
        batch_limit = batch_count if limit_eval_batches else None

        # Compute metrics
        self.metrics = self.compute_metrics(
            eval_set, batch_size=batch_size, batch_limit=batch_limit
        )

        return self

    def compute_metrics(
        self,
        dataset: ProbeDataset,
        batch_size: int = 32,
        batch_limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate metrics for the probes.

        Parameters
        ----------
        dataset : ProbeDataset
            Dataset of token IDs, activation masks, and labels
        batch_size : int
            Batch size
        batch_limit : None or int
            If None, limit batches

        Raises
        ------
        ValueError
            If predictions weren't collected for a probe
        """
        results = defaultdict(list)

        # Collect predictions
        for name, pred in self._iter_predictions(
            dataset, batch_size, batch_limit
        ):
            results[name].append(pred)

        # Calculate metrics
        metrics_df = []
        for probe in self.probes:
            name = probe.submodule
            if not results[name]:
                raise ValueError(f"No predictions collected for {name}")

            metrics = self._calculate_probe_metrics(probe, results[name])
            metrics_df.append(metrics)

        return pd.concat(metrics_df)

    def _iter_predictions(
        self,
        dataset: ProbeDataset,
        batch_size: int = 32,
        batch_limit: Optional[int] = None,
    ) -> Generator[tuple[str, BatchPredictions], None, None]:
        """Yield predictions batch by batch.

        Parameters
        ----------
        dataset : ProbeDataset
            Dataset of token IDs, activation masks, and labels
        batch_size : int
            Batch size
        batch_limit : None or int
            If None, limit batches

        Yields
        ------
        tuple[str, BatchPredictions]
            Tuple of (submodule_name, batch_predictions)
        """
        # Set up a DataLoader to yield out batches
        dataloader = DataLoader(dataset, batch_size=batch_size)

        # March through each batch and get features for the submodules
        for idx, batch in enumerate(dataloader):
            if batch_limit and idx >= batch_limit:
                break

            features = self.get_probe_features(
                batch["input_ids"], batch["attention_mask"]
            )
            labels = batch["labels"].numpy()

            for probe in self.probes:
                name = probe.submodule
                yield name, BatchPredictions(
                    preds=probe.predict(features[name]),
                    probs=probe.predict_proba(features[name]),
                    labels=labels,
                )

    def _calculate_probe_metrics(
        self, probe: LinearProbe, batches: list[BatchPredictions]
    ) -> pd.DataFrame:
        """Calculate metrics for a single probe from batch results.

        Parameters
        ----------
        probe : LinearProbe
            The probe
        batches : list[BatchPredictions]
            Batches of features

        Returns
        -------
        pd.DataFrame
            Overall metrics for all batches
        """
        # Adjust original labels for pooling
        if not self.pool:
            y = [batch.labels.reshape(-1) for batch in batches]
        else:
            y = [batch.labels for batch in batches]

        metrics = Metrics.compute(
            y=np.concatenate(y),
            preds=np.concatenate([batch.preds for batch in batches]),
            probs=np.concatenate([batch.probs for batch in batches]),
            classes=probe.classes,
        )

        return pd.DataFrame(metrics.to_dict(), index=[probe.submodule])
