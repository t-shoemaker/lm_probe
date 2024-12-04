from dataclasses import dataclass
from typing import Optional, Self

import torch
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from nnsight import LanguageModel
from torch.utils.data import DataLoader

from .probe import LinearProbe, ProbeConfig
from .extractor import FeatureExtractor
from .dataset import ProbeDataset
from .metrics import Metrics
from .utils import setup_logger


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
        self, model: LanguageModel, probe_configs: list[ProbeConfig]
    ) -> None:
        """Initialize the runner.

        Parameters
        ----------
        model : LanguageModel
            Langauge model
        prob_configs : list[ProbeConfig]
            Probe configurations
        """
        self.model: LanguageModel = model
        self.feature_extractor: FeatureExtractor = FeatureExtractor(model)
        self.probes: list[LinearProbe] = [
            LinearProbe(config) for config in probe_configs
        ]
        self.submodules: list[str] = [probe.submodule for probe in self.probes]
        self._probe_map = {
            probe.submodule: idx for idx, probe in enumerate(self.probes)
        }
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

        Returns
        -------
        dict[str, np.ndarray]
            Output dictionary of submodule : feature pairs

        Raises
        ------
        AssertionError
            If probe features are not 2- or 3-dimensional
        """
        features = self.feature_extractor(
            self.submodules,
            input_ids,
            attention_mask=attention_mask,
            pool=True,
        )

        return {
            submodule: feat.detach().cpu().numpy()
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
        batch_count = 0
        for epoch in range(num_epoch):
            dataloader = DataLoader(train_set, batch_size=batch_size)
            for batch in dataloader:
                if not any(probe.is_trainable for probe in self.probes):
                    break

                features = self.get_probe_features(
                    batch["input_ids"], batch["attention_mask"]
                )
                labels = batch["labels"].numpy()

                for probe in self.probes:
                    if not probe.is_trainable:
                        continue

                    probe.take_step(features[probe.submodule], labels)

                batch_count += 1

            if not any(probe.is_trainable for probe in self.probes):
                break

        self.logger.info(
            "Finished training %d probe(s). Computing metrics",
            len(self.probes),
        )  # type: ignore[attr-defined]

        batch_limit = batch_count if limit_eval_batches else None
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
        results: dict[str, list[BatchPredictions]] = {
            module: [] for module in self.submodules
        }

        dataloader = DataLoader(dataset, batch_size=batch_size)
        for idx, batch in enumerate(dataloader):
            if batch_limit and idx >= batch_limit:
                break

            features = self.get_probe_features(
                batch["input_ids"], batch["attention_mask"]
            )
            labels = batch["labels"].numpy()

            for probe in self.probes:
                name = probe.submodule
                results[name].append(
                    BatchPredictions(
                        preds=probe.predict(features[name]),
                        probs=probe.predict_proba(features[name]),
                        labels=labels,
                    )
                )

        metrics_df = []
        for probe in self.probes:
            name = probe.submodule
            probe_results = results[name]
            if not probe_results:
                raise ValueError(f"No predictions collected for {name}")

            metrics = Metrics.compute(
                y=np.concatenate([r.labels for r in probe_results]),
                preds=np.concatenate([r.preds for r in probe_results]),
                probs=np.concatenate([r.probs for r in probe_results]),
                classes=probe.classes,
            )
            metrics_df.append(pd.DataFrame(metrics.to_dict(), index=[name]))

        return pd.concat(metrics_df)
