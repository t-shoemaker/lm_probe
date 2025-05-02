from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from nnsight import LanguageModel


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
        self.remote: bool = remote
        self._cache: dict[str, torch.Tensor] = {}

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
        # Ensure we have what we need for pooling
        if pool and attention_mask is None:
            raise ValueError("Attention mask required for mean pooling")

        # Clear the cache if asked
        if not cache:
            self._cache.clear()

        # Set up a tracer with nnsight and march through each of the model
        # submodules to extract features
        features = {}
        with self.model.trace(input_ids, remote=self.remote):
            for name in submodules:
                feat = self.model.get(name).save()
                features[name] = feat

        # Extract the hidden states from each submodule feature set. This must
        # be done outside the context manager because nnsight won't have
        # executed the forward pass yet
        output = {}
        for name, feat in features.items():
            # Get the hidden state tensor. If the original Envoy output was a
            # tuple of tensors, we need to do a little special handling
            hs = feat[0] if not isinstance(feat.shape, torch.Size) else feat
            if hs.ndim not in (2, 3):
                raise ValueError(f"{name} has unexpected ndim={hs.ndim}")

            # Are we pooling?
            if pool and hs.ndim == 3:
                assert attention_mask is not None
                hs = mean_pool(hs, attention_mask.to(hs.device))

            output[name] = hs

        # Are we caching?
        if cache:
            self._cache = output.copy()

        return output

    def get_cached(self, name: str) -> torch.Tensor:
        """Get cached features by submodule name.

        Parameters
        ----------
        name : str
            Stringified submodule name

        Returns
        -------
        torch.Tensor
            The features

        Raises
        ------
        KeyError
            If there are no features cached for the submodule
        """
        try:
            return self._cache[name]
        except KeyError:
            raise KeyError(f"No cached features for {name}") from None

    def clear_cache(self):
        """Clear all stored features."""
        self._cache.clear()


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
