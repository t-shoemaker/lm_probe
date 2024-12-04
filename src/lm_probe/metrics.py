from enum import Enum
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


class MetricName(str, Enum):
    """Enumeration of available probe metrics."""

    LOSS = "loss"
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    MATTHEWS = "matthews"


@dataclass
class Metrics:
    """Container for classification metrics."""

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    matthews: float

    @classmethod
    def compute(
        cls,
        y: NDArray[np.integer],
        preds: NDArray[np.integer],
        probs: NDArray[np.floating],
        classes: NDArray[np.integer],
        average: str = "macro",
    ) -> Self:
        """Compute all metrics at once.

        Parameters
        ----------
        y : np.ndarray
            True labels
        preds : np.ndarray
            Predicted labels
        probs : np.ndarray
            Predicted probabilities
        classes : np.ndarray
            Possible class labels
        average : str
            Strategy for averaging labels

        Returns
        -------
        Metrics
            Container with computed metrics
        """
        return cls(
            loss=log_loss(y, probs, labels=classes),
            accuracy=accuracy_score(y, preds),
            precision=precision_score(
                y, preds, labels=classes, average=average, zero_division=0
            ),
            recall=recall_score(
                y, preds, labels=classes, average=average, zero_division=0
            ),
            f1=f1_score(
                y, preds, labels=classes, average=average, zero_division=0
            ),
            matthews=matthews_corrcoef(y, preds),
        )

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary.

        Returns
        -------
        metrics : dict[str, float]
            Dictionary of metric names and values
        """
        return {
            metric.value: getattr(self, metric.value) for metric in MetricName
        }
