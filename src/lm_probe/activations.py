import torch
from nnsight import LanguageModel
from nnsight.envoy import Envoy


def get_submodule(
    model: LanguageModel, submodule_name: str, sep: str = "."
) -> Envoy:
    """Get a submodule from the language model.

    Submodule are in the format <module><sep><module>...

    Parameters
    ----------
    model : LanguageModel
        The model
    submodule_name : str
        Stringified name of the submodule
    sep : str
        Separator character for submodule components

    Returns
    -------
    Envoy
        The submodule

    Raises
    ------
    AttributeError
        If the model does not have a requested submodule component
    """
    modules = submodule_name.split(sep)
    submodule = model
    while len(modules) > 0:
        module, *_ = modules
        if not hasattr(submodule, module):
            raise AttributeError(
                f"Couldn't access module '{module}' for {submodule_name}"
            )

        submodule = getattr(submodule, module)
        modules.pop(0)

    return submodule


@torch.no_grad()
def get_batch_features(
    input_ids: torch.Tensor,
    model: LanguageModel,
    submodules: list[str],
) -> dict[str, torch.Tensor]:
    """Get hidden state features from a submodule for a batch.

    Parameters
    ----------
    input_ids : torch.Tensor
        Batch of input IDs with shape (batch_size, num_doc, num_token)
    model : nnsight.LanguageModel
        The model from which to extract features
    submodules : list[str]
        The stringified submodule names

    Returns
    -------
    dict[str, torch.Tensor]
        The features with shape (batch_size, num_doc, num_feat)
    """
    features = {submodule: torch.empty(0) for submodule in submodules}
    with model.trace(input_ids):
        for module in submodules:
            features[module] = get_submodule(model, module).save()

    for module, module_features in features.items():
        # Features are either tuples of (hidden_states, heads, ...) or just
        # tensors
        shape = module_features.shape
        if isinstance(shape, torch.Size):
            continue

        features[module] = module_features[0]

    return features


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
