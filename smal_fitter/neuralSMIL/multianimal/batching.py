"""
Animal-axis tensor plumbing.

The multi-animal model keeps *one* shared backbone and evaluates ``N`` specimen
heads on the resulting features.  Two shapes therefore appear everywhere:

``(B, N, ...)``
    "animal-major" — the natural shape of a multi-animal prediction.

``(B * N, ...)``
    "flattened" — what every existing single-animal module (SMAL forward,
    renderer, loss) already accepts.  Flattening the animal axis into the batch
    axis is exactly the "batched operation rather than three fundamentally
    different SMAL models" the design doc asks for (§7), and it is what lets the
    entire existing machinery be reused untouched.

The helpers here are intentionally tiny and pure so they can be unit-tested
without a GPU, a SMAL model or pytorch3d.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch


def flatten_animal_axis(tensor: torch.Tensor) -> torch.Tensor:
    """``(B, N, ...) -> (B * N, ...)`` in animal-minor (row-major) order.

    Row-major means sample 0's specimens come first::

        [b0a0, b0a1, ..., b0a{N-1}, b1a0, ...]

    :func:`unflatten_animal_axis` is the exact inverse.
    """
    if tensor.dim() < 2:
        raise ValueError(f"expected at least 2 dims (B, N, ...), got shape {tuple(tensor.shape)}")
    b, n = tensor.shape[0], tensor.shape[1]
    return tensor.reshape(b * n, *tensor.shape[2:])


def unflatten_animal_axis(tensor: torch.Tensor, num_animals: int) -> torch.Tensor:
    """``(B * N, ...) -> (B, N, ...)``; inverse of :func:`flatten_animal_axis`."""
    if num_animals < 1:
        raise ValueError(f"num_animals must be >= 1, got {num_animals}")
    total = tensor.shape[0]
    if total % num_animals != 0:
        raise ValueError(f"leading dim {total} is not divisible by num_animals={num_animals}")
    return tensor.reshape(total // num_animals, num_animals, *tensor.shape[1:])


def expand_scene_to_animals(tensor: torch.Tensor, num_animals: int) -> torch.Tensor:
    """``(B, ...) -> (B * N, ...)`` by repeating each sample ``N`` times.

    Used to broadcast *scene-level* quantities (camera parameters, image
    features) across the specimens of the same scene, in the same row-major
    order :func:`flatten_animal_axis` produces.
    """
    if num_animals < 1:
        raise ValueError(f"num_animals must be >= 1, got {num_animals}")
    return tensor.unsqueeze(1).expand(-1, num_animals, *([-1] * (tensor.dim() - 1))).reshape(
        tensor.shape[0] * num_animals, *tensor.shape[1:]
    )


def stack_specimen_params(
    per_specimen: Sequence[Dict[str, torch.Tensor]],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Stack ``N`` per-specimen parameter dicts into ``(B, N, ...)`` tensors.

    Only keys present in *every* specimen dict (and tensor-valued) are stacked;
    anything else is skipped, so auxiliary entries such as ``iteration_history``
    do not break the stacking.
    """
    if not per_specimen:
        return {}
    candidate_keys = list(keys) if keys is not None else list(per_specimen[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for key in candidate_keys:
        values = [params.get(key) for params in per_specimen]
        if any(v is None or not isinstance(v, torch.Tensor) for v in values):
            continue
        out[key] = torch.stack(values, dim=1)
    return out


def select_specimen(stacked: Dict[str, torch.Tensor], specimen_index: int) -> Dict[str, torch.Tensor]:
    """Slice ``(B, N, ...)`` parameter tensors down to one specimen ``(B, ...)``."""
    return {key: value[:, specimen_index] for key, value in stacked.items() if isinstance(value, torch.Tensor)}


def animal_mask_to_tensor(
    masks: Sequence[Sequence[bool]],
    num_animals: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the ``(B, N)`` boolean presence mask from per-sample masks.

    Samples that declare fewer than ``num_animals`` specimens are padded with
    ``False``, which is how a batch mixing 2-mouse and 3-mouse clips is made
    rectangular.
    """
    rows: List[List[bool]] = []
    for mask in masks:
        row = [bool(v) for v in list(mask)[:num_animals]]
        row.extend([False] * (num_animals - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.bool, device=device)


def masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean of ``values`` over the entries selected by ``mask``.

    Returns a differentiable zero (rather than NaN) when nothing is selected, so
    a batch in which a specimen slot is entirely absent contributes no gradient
    instead of poisoning the loss.
    """
    if mask is None:
        return values.mean() if values.numel() > 0 else torch.zeros((), device=values.device, dtype=values.dtype)
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum()
    if float(denom.detach()) <= 0.0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * mask_f).sum() / (denom + eps)


def aggregate_specimen_losses(
    losses: Sequence[torch.Tensor],
    counts: Sequence[int],
    reduction: str = "mean",
) -> torch.Tensor:
    """Aggregate per-specimen scalar losses into the sample loss.

    Args:
        losses: One scalar per specimen slot.
        counts: Number of batch entries in which that specimen was present.
            Slots with a zero count are dropped entirely so an absent specimen
            neither dilutes nor inflates the mean.
        reduction: ``"mean"`` (default) averages the present specimens equally —
            matching design doc §9's "aggregate/average across animals".
            ``"sum"`` adds them.  ``"weighted_mean"`` weights each specimen by
            how often it was present, which keeps the gradient scale stable when
            specimens appear at very different rates.

    Returns:
        A scalar tensor.  When no specimen was present anywhere the result is a
        differentiable zero.
    """
    if not losses:
        raise ValueError("aggregate_specimen_losses() requires at least one specimen loss")
    if len(losses) != len(counts):
        raise ValueError(f"losses ({len(losses)}) and counts ({len(counts)}) must have the same length")

    device = losses[0].device
    dtype = losses[0].dtype
    present = [(loss, count) for loss, count in zip(losses, counts) if count > 0]
    if not present:
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True) + sum(
            loss * 0.0 for loss in losses
        )

    if reduction == "sum":
        return torch.stack([loss for loss, _ in present]).sum()
    if reduction == "weighted_mean":
        weights = torch.tensor([float(count) for _, count in present], device=device, dtype=dtype)
        weights = weights / weights.sum()
        return (torch.stack([loss for loss, _ in present]) * weights).sum()
    if reduction == "mean":
        return torch.stack([loss for loss, _ in present]).mean()
    raise ValueError(f"unknown reduction '{reduction}' (expected 'mean', 'sum' or 'weighted_mean')")


def merge_loss_components(
    per_specimen_components: Sequence[Dict[str, torch.Tensor]],
    counts: Sequence[int],
    specimen_ids: Optional[Sequence[str]] = None,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Merge per-specimen loss-component dicts into one reporting dict.

    The merged dict keeps the original component names (averaged over present
    specimens) so existing logging, plotting and checkpoint code needs no
    changes, and additionally exposes ``"<component>/<specimen_id>"`` entries so
    a specimen that is failing to converge is visible in the training logs.
    """
    merged: Dict[str, torch.Tensor] = {}
    if not per_specimen_components:
        return merged

    ids = list(specimen_ids) if specimen_ids is not None else [f"specimen_{i}" for i in range(len(per_specimen_components))]

    all_keys: List[str] = []
    for components in per_specimen_components:
        for key in components:
            if key not in all_keys:
                all_keys.append(key)

    for key in all_keys:
        values: List[torch.Tensor] = []
        value_counts: List[int] = []
        for idx, components in enumerate(per_specimen_components):
            if key not in components:
                continue
            value = components[key]
            if not isinstance(value, torch.Tensor):
                continue
            name = ids[idx] if idx < len(ids) else f"specimen_{idx}"
            merged[f"{key}/{name}"] = value.detach()
            values.append(value)
            value_counts.append(counts[idx] if idx < len(counts) else 0)
        if values:
            merged[key] = aggregate_specimen_losses(values, value_counts, reduction=reduction)
    return merged
