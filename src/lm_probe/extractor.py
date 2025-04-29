from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from nnsight import LanguageModel


@dataclass
class SubmoduleFeatures:
    """Features from a submodule."""

    name: str
    features: torch.Tensor

    @property
    def is_tuple_output(self) -> bool:
        """Check if the original Envoy output was a tuple of tensors."""
        return not isinstance(self.features.shape, torch.Size)

    def get_hidden_state(self) -> torch.Tensor:
        """Get the hidden state tensor."""
        return self.features[0] if self.is_tuple_output else self.features


class FeatureExtractor:
    """Extractor for a LanguageModel's submodule features."""

    def __init__(self, model: "LanguageModel", remote: bool = False) -> None:
        """Initialize the extractor.

        Parameters
        ----------
        model : LanguageModel
            The model
        remote : bool
            Whether to run the model remotely on NDIF (requires API key)
        """
        self.model: "LanguageModel" = model
        self.features: dict[str, SubmoduleFeatures] = {}
        self.remote: bool = remote

    def __repr__(self) -> str:
        """Class repr."""
        return f"Extractor for {self.model.config.model_type}"

    @torch.no_grad()
    def __call__(
        self,
        submodules: list[str],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pool: bool = True,
        cache: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract hidden state features from submodules.

        Parameters
        ----------
        submodules : list[str]
            The stringified submodule names
        input_ids : torch.Tensor
            Batch of input IDs with shape (batch_size, num_doc, num_token)
        attention_mask : torch.Tensor, optional
            Attention mask for pooling, required for pooling
        pool : bool
            Whether to mean pool 3D features
        cache : bool
            Whether to cache features for the input_ids

        Returns
        -------
        dict[str, torch.Tensor]
            The features with shape (batch_size, num_tok, num_feat) or
            (batch_size, num_feat)

        Raises
        ------
        ValueError
            If pool is True but attention_mask is None
            If features aren't 2- or 3-dimensional
        """
        if pool and attention_mask is None:
            raise ValueError("Attention mask required for mean pooling")

        # Set up a tracer with nnsight and march through each of the model
        # submodules to extract features
        with self.model.trace(input_ids, remote=self.remote):
            for submodule_name in submodules:
                feat = self.model.get(submodule_name).save()
                self.features[submodule_name] = SubmoduleFeatures(
                    name=submodule_name, features=feat
                )

        # Pluck the hidden states from each submodule feature set. This must be
        # done outside the context manager because nnsight won't have executed
        # the forward pass yet
        features = {
            submodule_name: feat.get_hidden_state()
            for submodule_name, feat in self.features.items()
        }

        # Are we providing features for every token?
        if not pool:
            return features

        # Are we keeping features?
        if not cache:
            self.clear_features()

        # Do mean pooling on the features
        pooled_features = {}
        for submodule_name, feat in features.items():
            ndim = feat.ndim
            if not (1 < ndim < 4):
                raise ValueError(
                    f"Features for {submodule_name} are {feat.shape}"
                )

            if ndim == 3:
                assert attention_mask is not None
                feat = mean_pool(feat, attention_mask.to(feat.device))

            pooled_features[submodule_name] = feat

        return pooled_features

    def get_features(self, submodule_name: str) -> Optional[SubmoduleFeatures]:
        """Get features by submodule name.

        Parameters
        ----------
        submodule_name : str
            Stringified submodule name

        Returns
        -------
        SubmoduleFeatures, optional
            The features

        Raises
        ------
        ValueError
            If there are no features cached for the submodule
        """
        if not self.features:
            raise ValueError(f"No features cached for {submodule_name}")

        return self.features.get(submodule_name)

    def clear_features(self):
        """Clear all stored features."""
        self.features.clear()


def mean_pool(
    features: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling across model features.

    This is based on the mean pooling implementation in SBERT.
        SBERT: https://github.com/UKPLab/sentence-transformers

    Parameters
    ----------
    features : torch.Tensor
        Features with the shape (batch_size, seq_len, num_feat)
    attention_mask : torch.Tensor
        Attention mask for the tokens with the shape (batch_size, seq_len)

    Returns
    -------
    torch.Tensor
        Pooled embeddings with the shape (batch_size, num_dim)
    """
    mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
    sum_features = torch.sum(features * mask, 1)
    sum_mask = mask.sum(1).clamp(min=1e-8)

    return sum_features / sum_mask
