from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
from typing import Any, Optional, Self
from functools import partial

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from .control import EarlyStopping
from .metrics import Metrics
from .utils import setup_logger


@dataclass
class ProbeConfig:
    """Linear probe configuration.

    Parameters
    ----------
    submodule : str
        Stringified name of the submodule
    classes : Iterable[int]
        Class labels
    test_size : float
        Testing size for early early stopping
    max_steps : int
        Maximum number of update steps
    early_stopping : bool
        Whether to do early stopping
    patience : int
        How many evaluation periods to wait before triggering early stopping
    threshold : float
        Threshold for score equivalence with early stopping
    warmup_steps : int
        Number of steps to wait before checking for early stopping
    pool: bool
        Whether to train on pooled (document-level) features
    null_label : int or None
        A null label value, which masks out token-level features
    model_kwargs : dict[str, Any]
        Model keywords
    """

    submodule: str
    classes: Iterable[int]
    test_size: float = 0.1
    max_steps: int = 1_000
    early_stopping: bool = True
    patience: int = 5
    threshold: float = 1e-5
    warmup_steps: int = 10
    pool: bool = True
    null_label: Optional[int] = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class LinearProbe:
    """Linear probe classifier for online learning on model features."""

    logger = setup_logger(__name__)

    def __init__(self, config) -> None:
        """Initialize the probe.

        Parameters
        ----------
        config : ProbeConfig
            Probe configuration
        """
        # Initialize the model
        self.model: SGDClassifier = SGDClassifier(
            loss="log_loss", **config.model_kwargs
        )
        self.scaler: StandardScaler = StandardScaler()

        # Set up probe info
        self.submodule: str = config.submodule
        self.classes: NDArray[np.integer] = np.fromiter(config.classes, int)
        self.pool: bool = config.pool
        self.test_size: float = config.test_size
        self.tt_split: Callable = partial(
            train_test_split, test_size=config.test_size
        )

        # Set up training info
        self.max_steps: int = config.max_steps
        self.early_stopping: bool = config.early_stopping
        self.warmup_steps: int = config.warmup_steps
        self.null_label: Optional[int] = config.null_label
        self.stopper: EarlyStopping = EarlyStopping(
            config.patience, False, config.threshold
        )

        # Set up training metadata
        self.is_trainable: bool = True
        self.stopped_early: bool = False
        self.steps_taken: int = 0
        self.best_train_loss: float = np.inf
        self.metrics: Optional[Metrics] = None

    def __repr__(self) -> str:
        """Repr method.

        Returns
        -------
        str
            Basic info about the probe
        """
        return f"Probe for {self.submodule}"

    def take_step(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
    ) -> Self:
        """Take an update step.

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray
            Feature labels

        Returns
        -------
        LinearProbe
            The updated probe
        """
        if not self.is_trainable:
            self.logger.warning("Probe is no longer trainable")  # type: ignore[attr-defined]
            return self

        # Are we dealing with token-level information (i.e., a FeatureExtractor
        # didn't pool the model representations)?
        if not self.pool:
            X, y = self._flatten(X, y, self.null_label)

        # Trainable probes take two kinds of incremental steps: one that fits
        # on the entirey of a batch's data and one that does a train/test split
        match (self.early_stopping, self.steps_taken > self.warmup_steps):
            # If we aren't doing early stopping or we haven't reached our
            # warmup stems, fit on the whole batch
            case (False, _) | (True, False):
                self.scaler.partial_fit(X)
                X = self.scaler.transform(X)

                self.model.partial_fit(X, y, classes=self.classes)
                self._log_step(X, y)

            case (True, True):
                # Otherwise, do a train/test split
                X_train, X_test, y_train, y_test = self.tt_split(X, y)

                self.scaler.partial_fit(X_train)
                X_train = self.scaler.transform(X_train)
                X_test = self.scaler.transform(X_test)

                self.model.partial_fit(X_train, y_train, classes=self.classes)
                self._log_step(X_test, y_test)

                # If early stopping is set, we check whether it's time to stop
                if self.is_trainable:
                    self._check_early_stopping(X_test, y_test)

        return self

    def _flatten(
        self,
        X: NDArray[np.floating],
        y: Optional[NDArray[np.integer]] = None,
        null_label: Optional[int] = None,
    ) -> tuple[NDArray[np.floating], Optional[NDArray[np.integer]]]:
        """Flatten token-by-token feature/label arrays.

        The `null_label` value should correspond to tokens that the probe will
        not be trained on. It's set in `ProbeConfig`

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray or None
            Feature labels
        null_label : int or None
            Null label value

        """
        if not null_label and self.steps_taken == 0:
            self.logger.warn(
                (
                    "Probe %s does not have a null_value. It will train on "
                    "features for every token. If this is intended behavior, "
                    "this message can be ignored. If not, set null_value in "
                    "the ProbeConfig"
                ),
                self.submodule,
            )  # type: ignore[attr-defined]

        # Flatten the fatures. If we only have those, we're done
        *_, num_feat = X.shape
        X_flat = X.reshape(-1, num_feat)
        if (null_label is None) and (y is None):
            return X_flat, None

        # Otherwise, drop tokens outside the label target
        y_flat = y.reshape(-1)
        (mask,) = np.where(y_flat != self.null_label)

        return X_flat[mask], y_flat[mask]

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Predict classes.

        Parameters
        ----------
        X : np.ndarray
            Model features

        Returns
        -------
        np.ndarray
            Class predictions
        """
        if not self.pool:
            X, _ = self._flatten(X, None, None)

        scaled = self.scaler.transform(X)
        preds = self.model.predict(scaled)

        return preds

    def predict_proba(
        self, X: NDArray[np.floating], eps: float = 1e-10
    ) -> NDArray[np.floating]:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            Model features
        eps : float
            Epsilon offset for numerical stability

        Returns
        -------
        np.ndarray
            Probabilities for each class with shape (batch_size, num_class)
        """
        if not self.pool:
            X, _ = self._flatten(X, None, None)

        scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(scaled)
        probs = np.clip(probs, eps, 1.0 - eps)

        return probs

    def score(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> float:
        """Calculate cross-entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray
            Feature labels

        Returns
        -------
        float
            Cross-entropy loss
        """
        probs = self.predict_proba(X)
        loss = log_loss(y, probs, labels=self.classes)

        return loss

    def _stop_info(self, message: str) -> None:
        """Log a stopping message.

        Parameters
        ----------
        message : str
            The stopping message with format entries for steps_taken and the
            submodule
        """
        self.logger.info(message, self.steps_taken, self.submodule)  # type: ignore[attr-defined]

    def _log_step(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> None:
        """Log a step.

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray
            Feature labels
        """
        self.steps_taken += 1
        self.is_trainable = self.steps_taken < self.max_steps

        if not self.is_trainable:
            self._stop_info("Probe completed after %d steps for %s")

    def _check_early_stopping(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> None:
        """Check for early stopping.

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray
            Feature labels
        """
        score = self.score(X, y)
        self.best_train_loss = np.min([score, self.best_train_loss])
        self.is_trainable = not self.stopper(score)
        self.stopped_early = not self.is_trainable

        if not self.is_trainable:
            self._stop_info("Early stopping after %d steps for %s")

    def compute_metrics(
        self, X: NDArray[np.floating], y: NDArray[np.integer]
    ) -> pd.DataFrame:
        """Generate metrics for predictions.

        Parameters
        ----------
        X : np.ndarray
            Model features
        y : np.ndarray
            Feature labels

        Returns
        -------
        pd.DataFrame
            A metrics DataFrame
        """
        probs = self.predict_proba(X)
        preds = self.predict(X)
        self.metrics = Metrics.compute(
            y=y, preds=preds, probs=probs, classes=self.classes
        )

        return self.metrics
