import numpy as np


class EarlyStopping:
    """Handler for early stopping conditions."""

    def __init__(
        self,
        patience: int = 5,
        greater_is_better: bool = False,
        threshold: float = 1e-8,
    ) -> None:
        """Initialize the handler.

        Parameters
        ----------
        patience : int
            How many iterations to wait before triggering early stopping
        greater_is_better : bool
            Should a candidate score be higher or lower than current score?
        threshold : float
            Threshold for score equivalence
        """
        self.patience: int = patience
        self.greater_is_better: bool = greater_is_better
        self.best_score: float = -np.inf if greater_is_better else np.inf
        self.threshold: float = threshold
        self.counter: int = 0

    def __call__(self, score: float) -> bool:
        """When handler is called, determine early stopping condition.

        Parameters
        ----------
        score : float
            The candidate score

        Returns
        -------
        bool
            Whether early stopping has been reached
        """
        close_scores = np.isclose(score, self.best_score, rtol=self.threshold)
        if self.greater_is_better:
            has_improved = score > self.best_score and not close_scores
        else:
            has_improved = score < self.best_score and not close_scores

        self.counter = 0 if has_improved else self.counter + 1
        self.best_score = score if has_improved else self.best_score

        return self.counter >= self.patience
