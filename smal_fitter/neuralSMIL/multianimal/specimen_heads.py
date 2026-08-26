"""
The ``N`` specimen parameter heads.

Design doc §3-§5: keep the shared backbone, add ``N - 1`` copies of the existing
parameter head, and bind head ``i`` permanently to specimen ``i``.  Two
realisations of "N heads" are provided behind one interface:

``replicated``
    Literally ``N`` independent heads, each a fresh copy of the single-animal
    head.  This is what the design document specifies, and it is the default.
    Cost: ``N ×`` head parameters (noticeable for the transformer decoder head).

``shared_query``
    One head plus ``N`` learned *specimen embeddings*.  The embedding is added
    to the pooled feature vector and injected as an extra cross-attention
    context token, so the single set of head weights is conditioned on which
    specimen it is currently decoding.  Cheaper, scales to more specimens, and
    shares statistical strength across animals of the same species — but it is a
    deviation from the written design, so it is opt-in.

Both strategies preserve the strict head ↔ specimen correspondence that removes
the need for Hungarian matching or a permutation-invariant loss: the specimen
index is an *input*, never something the model discovers.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .batching import expand_scene_to_animals, unflatten_animal_axis

HeadFactory = Callable[[], nn.Module]

#: Strategy names accepted by :func:`build_specimen_heads`.
REPLICATED = "replicated"
SHARED_QUERY = "shared_query"
STRATEGIES = (REPLICATED, SHARED_QUERY)


class SpecimenHeads(nn.Module, ABC):
    """Base class for a bank of ``N`` identity-bound parameter heads.

    Subclasses implement :meth:`forward`, returning one parameter dict per
    specimen slot, in slot order.  The contract is deliberately narrow so the
    regressors do not care which strategy is in use.
    """

    def __init__(self, num_animals: int):
        super().__init__()
        if num_animals < 1:
            raise ValueError(f"num_animals must be >= 1, got {num_animals}")
        self.num_animals = int(num_animals)

    @property
    @abstractmethod
    def strategy(self) -> str:
        """Name of the realisation strategy (``'replicated'`` / ``'shared_query'``)."""

    @abstractmethod
    def forward(
        self,
        global_features: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode every specimen from the shared image features.

        Args:
            global_features: ``(B, feature_dim)`` pooled features.
            spatial_features: Optional ``(B, S, context_dim)`` spatial tokens for
                cross-attention (ViT patch tokens, per-view fused tokens, ...).

        Returns:
            A list of ``N`` parameter dicts, each with ``(B, ...)`` tensors.
        """

    @abstractmethod
    def load_single_animal_head_state(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Initialise every specimen head from one pretrained single-animal head.

        This is design doc §3's "new heads can be initialized by copying the
        pretrained single-animal head weights", and it is what makes a
        multi-animal run resumable from an existing single-animal checkpoint.
        """

    def describe(self) -> str:
        """One-line human-readable summary, for training logs."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"{type(self).__name__}(strategy={self.strategy}, N={self.num_animals}, params={trainable:,})"


class ReplicatedSpecimenHeads(SpecimenHeads):
    """``N`` fully independent parameter heads (the design-document default).

    Args:
        head_factory: Zero-argument callable producing one fresh head.  Use
            :func:`~smal_fitter.neuralSMIL.multianimal.heads.make_head_factory`
            so the heads are built by the regressor's own construction routine
            and can never drift from the single-animal head.
        num_animals: Number of specimen slots ``N``.
        tie_first_head_init: When True (default) all heads start from the *same*
            random initialisation, matching "new heads copied from the existing
            head" even when training from scratch.  Set False to give every head
            an independent initialisation.
    """

    def __init__(self, head_factory: HeadFactory, num_animals: int, tie_first_head_init: bool = True):
        super().__init__(num_animals)
        first = head_factory()
        heads = [first]
        for _ in range(self.num_animals - 1):
            heads.append(copy.deepcopy(first) if tie_first_head_init else head_factory())
        self.heads = nn.ModuleList(heads)

    @property
    def strategy(self) -> str:
        return REPLICATED

    def forward(
        self,
        global_features: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        # Independent weights mean there is nothing to batch across specimens;
        # N is small (2-3 in practice) and the heads are cheap next to the
        # backbone, so a plain loop is both clearest and fast enough.
        return [head(global_features, spatial_features) for head in self.heads]

    def load_single_animal_head_state(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        for head in self.heads:
            _load_head_state(head, state_dict, strict=strict)


class SharedQuerySpecimenHeads(SpecimenHeads):
    """One head conditioned on a learned per-specimen embedding.

    The embedding enters the head twice, mirroring how the multi-view code
    already injects view identity:

    * added to the pooled feature vector, and
    * prepended to the cross-attention context as a dedicated specimen token
      (plus added to every context token, so spatial attention is also
      specimen-aware).

    All ``N`` specimens are decoded in a **single** head call by folding the
    animal axis into the batch axis, so this strategy costs one head forward
    over ``B × N`` rows rather than ``N`` forwards over ``B``.

    Args:
        head_factory: Builds the single shared head.
        num_animals: Number of specimen slots ``N``.
        feature_dim: Width of ``global_features``.
        context_dim: Width of ``spatial_features``; defaults to ``feature_dim``.
        embedding_std: Init std of the embeddings.  Small by design: at init the
            head must behave like the single-animal head it was copied from.
    """

    def __init__(
        self,
        head_factory: HeadFactory,
        num_animals: int,
        feature_dim: int,
        context_dim: Optional[int] = None,
        embedding_std: float = 0.02,
    ):
        super().__init__(num_animals)
        self.head = head_factory()
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim if context_dim is not None else feature_dim)

        self.specimen_embedding = nn.Embedding(self.num_animals, self.feature_dim)
        self.context_embedding = nn.Embedding(self.num_animals, self.context_dim)
        nn.init.normal_(self.specimen_embedding.weight, mean=0.0, std=embedding_std)
        nn.init.normal_(self.context_embedding.weight, mean=0.0, std=embedding_std)

    @property
    def strategy(self) -> str:
        return SHARED_QUERY

    def forward(
        self,
        global_features: torch.Tensor,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        batch_size = global_features.shape[0]
        n = self.num_animals
        device = global_features.device

        # (N,) -> (B*N,) in the same animal-minor order as expand_scene_to_animals
        specimen_index = torch.arange(n, device=device).repeat(batch_size)

        feats = expand_scene_to_animals(global_features, n) + self.specimen_embedding(specimen_index)

        context = None
        if spatial_features is not None:
            context = expand_scene_to_animals(spatial_features, n)  # (B*N, S, C)
            context_embed = self.context_embedding(specimen_index).unsqueeze(1)  # (B*N, 1, C)
            context = torch.cat([context_embed, context + context_embed], dim=1)  # (B*N, 1+S, C)

        flat_params = self.head(feats, context)
        return _split_flat_params(flat_params, n)

    def load_single_animal_head_state(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        _load_head_state(self.head, state_dict, strict=strict)


def build_specimen_heads(
    strategy: str,
    head_factory: HeadFactory,
    num_animals: int,
    feature_dim: int,
    context_dim: Optional[int] = None,
    tie_first_head_init: bool = True,
) -> SpecimenHeads:
    """Construct the configured :class:`SpecimenHeads` implementation.

    Raises:
        ValueError: on an unknown strategy name, naming the valid options.
    """
    if strategy == REPLICATED:
        return ReplicatedSpecimenHeads(head_factory, num_animals, tie_first_head_init=tie_first_head_init)
    if strategy == SHARED_QUERY:
        return SharedQuerySpecimenHeads(
            head_factory,
            num_animals,
            feature_dim=feature_dim,
            context_dim=context_dim,
        )
    raise ValueError(f"unknown specimen-head strategy '{strategy}' (expected one of {STRATEGIES})")


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _split_flat_params(params: Dict[str, object], num_animals: int) -> List[Dict[str, torch.Tensor]]:
    """Split a ``(B*N, ...)`` parameter dict into ``N`` dicts of ``(B, ...)``.

    Non-tensor entries (the transformer head's ``iteration_history``) are split
    element-wise where possible and otherwise carried through unchanged on every
    specimen, so downstream diagnostics keep working.
    """
    out: List[Dict[str, torch.Tensor]] = [{} for _ in range(num_animals)]
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            unflat = unflatten_animal_axis(value, num_animals)  # (B, N, ...)
            for i in range(num_animals):
                out[i][key] = unflat[:, i]
        elif isinstance(value, dict):
            for i in range(num_animals):
                out[i][key] = {
                    sub_key: [
                        unflatten_animal_axis(item, num_animals)[:, i] if isinstance(item, torch.Tensor) else item
                        for item in sub_value
                    ]
                    if isinstance(sub_value, (list, tuple))
                    else sub_value
                    for sub_key, sub_value in value.items()
                }
        else:
            for i in range(num_animals):
                out[i][key] = value
    return out


def _load_head_state(head: nn.Module, state_dict: Mapping[str, torch.Tensor], strict: bool) -> None:
    """Load a head-local state dict, tolerating the extracted-MLP key mapping."""
    load_inline = getattr(head, "load_inline_head_state", None)
    if callable(load_inline) and not _looks_like_own_state(head, state_dict):
        load_inline(state_dict, strict=strict)
        return
    missing, unexpected = head.load_state_dict(dict(state_dict), strict=False)
    if strict and (missing or unexpected):
        raise KeyError(
            f"specimen head state mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _looks_like_own_state(head: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> bool:
    """True when ``state_dict`` already uses this module's own key names."""
    own_keys = set(head.state_dict().keys())
    return bool(own_keys) and len(own_keys.intersection(state_dict.keys())) >= max(1, len(own_keys) // 2)


def specimen_parameter_groups(
    heads: SpecimenHeads,
    base_lr: float,
    per_specimen_lr_scale: Optional[Sequence[float]] = None,
) -> List[Dict[str, object]]:
    """Optimiser parameter groups, optionally with a per-specimen LR scale.

    Useful when one specimen is systematically harder (e.g. the mouse that is
    occluded most often) and should be given a different learning rate.  With
    ``per_specimen_lr_scale=None`` a single group at ``base_lr`` is returned.
    """
    if per_specimen_lr_scale is None or not isinstance(heads, ReplicatedSpecimenHeads):
        return [{"params": list(heads.parameters()), "lr": base_lr}]

    if len(per_specimen_lr_scale) != heads.num_animals:
        raise ValueError(
            f"per_specimen_lr_scale has {len(per_specimen_lr_scale)} entries, expected {heads.num_animals}"
        )
    return [
        {"params": list(head.parameters()), "lr": base_lr * float(scale)}
        for head, scale in zip(heads.heads, per_specimen_lr_scale)
    ]
