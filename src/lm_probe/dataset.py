import torch
from typing import TYPE_CHECKING

from torch.utils.data import Dataset, random_split

if TYPE_CHECKING:
    from torch.utils.data import Subset


class ProbeDataset(Dataset):
    """Dataset for probes."""

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token input IDs with shape (num_doc, seq_len)
        attention_mask torch.Tensor
            Attention mask for the tokens with shape (num_doc, seq_len)
        labels : torch.Tensor or list
            Labels

        Raises
        ------
        ValueError
            If the input IDs and attention mask shapes mismatch
        """
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"Input IDs and attention mask shapes mismatch: "
                f"got {input_ids.shape=} and {attention_mask.shape=}"
            )

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = torch.as_tensor(labels, dtype=torch.uint8)

    def __len__(self) -> int:
        """Dataset size."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get item from the dataset.

        Parameters
        ----------
        idx : int
            Index

        Returns
        -------
        dict[str, torch.Tensor]
            Item at index `idx`
        """
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def tt_split(self, test_size: float = 0.2) -> tuple["Subset", "Subset"]:
        """Do a random train/test split.

        Parameters
        ----------
        test_size : float
            Test size

        Returns
        -------
        tuple[Subset]
            Train/test datasets
        """
        test_size = round(test_size * len(self))
        train_size = len(self) - test_size
        train_set, test_set = random_split(self, [train_size, test_size])

        return train_set, test_set
