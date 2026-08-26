"""
Per-specimen loss aggregation.

Design doc §9: because head ``i`` is permanently bound to specimen ``i``, the
loss is simply::

    L = L(specimen_1) + L(specimen_2) + ... + L(specimen_N)

with the *existing* SMILify losses applied independently to each specimen and no
permutation matching anywhere.  This module owns the two things that are not
quite that simple:

1. **Scene-level terms must not be counted N times.**  Camera parameters belong
   to the view, not to an animal (§6), so their supervision is evaluated for
   specimen 0 only; every later specimen gets those weights zeroed.

2. **Absent specimens must not dilute the mean.**  A slot that is not present in
   any sample of a batch contributes no gradient and is dropped from the
   average, so a batch of 2-mouse clips in an ``N = 3`` run trains exactly like
   a 2-mouse run.

The aggregator never touches the loss maths itself: it takes a callable that
computes *one* specimen's loss with the existing single-animal machinery.  That
keeps every tested loss path shared between the single- and multi-animal models.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .batching import aggregate_specimen_losses, merge_loss_components
from .schema import SCENE_LEVEL_LOSS_KEYS

#: ``loss_fn(specimen_index, loss_weights) -> (loss, components)``
SpecimenLossFn = Callable[[int, Dict[str, float]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]


def weights_for_specimen(
    loss_weights: Dict[str, float],
    specimen_index: int,
    scene_loss_keys: Sequence[str] = SCENE_LEVEL_LOSS_KEYS,
) -> Dict[str, float]:
    """Return the loss weights to use when evaluating one specimen.

    Scene-level components (camera FOV / rotation / translation) are supervised
    once per sample, on specimen 0; for every other specimen their weight is set
    to zero.  Without this the camera loss would be multiplied by ``N`` relative
    to the body losses and silently rebalance the whole curriculum.

    The input dict is never mutated.
    """
    weights = dict(loss_weights)
    if specimen_index != 0:
        for key in scene_loss_keys:
            if key in weights:
                weights[key] = 0.0
    return weights


class MultiAnimalLossAggregator:
    """Evaluates and combines the per-specimen losses of one batch.

    Args:
        num_animals: Number of specimen slots ``N``.
        specimen_ids: Stable identity labels, used to name per-specimen metrics.
        reduction: ``"mean"`` (default), ``"sum"`` or ``"weighted_mean"``; see
            :func:`~smal_fitter.neuralSMIL.multianimal.batching.aggregate_specimen_losses`.
        scene_loss_keys: Component names that are scene-level, not animal-level.
        drop_absent: Skip specimen slots with a zero presence count entirely
            rather than evaluating a guaranteed-zero loss for them.
    """

    def __init__(
        self,
        num_animals: int,
        specimen_ids: Optional[Sequence[str]] = None,
        reduction: str = "mean",
        scene_loss_keys: Sequence[str] = SCENE_LEVEL_LOSS_KEYS,
        drop_absent: bool = True,
    ):
        if num_animals < 1:
            raise ValueError(f"num_animals must be >= 1, got {num_animals}")
        self.num_animals = int(num_animals)
        self.specimen_ids = (
            list(specimen_ids) if specimen_ids else [f"specimen_{i}" for i in range(self.num_animals)]
        )
        if len(self.specimen_ids) != self.num_animals:
            raise ValueError(
                f"specimen_ids has {len(self.specimen_ids)} entries, expected num_animals={self.num_animals}"
            )
        self.reduction = reduction
        self.scene_loss_keys = tuple(scene_loss_keys)
        self.drop_absent = bool(drop_absent)

    def __call__(
        self,
        loss_fn: SpecimenLossFn,
        loss_weights: Dict[str, float],
        presence_counts: Sequence[int],
        device: Optional[torch.device] = None,
        return_components: bool = False,
    ):
        """Evaluate every specimen and combine the results.

        Args:
            loss_fn: Computes one specimen's ``(loss, components)`` using the
                existing single-animal loss.  It receives the specimen index and
                the already-adjusted loss weights.
            loss_weights: Base loss weights for this epoch.
            presence_counts: ``(N,)`` — how many batch entries contain each
                specimen.  Slots with zero are dropped (see ``drop_absent``).
            device: Device for the zero-loss fallback when nothing is present.
            return_components: Also return the merged per-component dict.

        Returns:
            ``loss`` or ``(loss, components)``.  ``components`` carries both the
            averaged component names (so existing logging keeps working) and
            ``"<component>/<specimen_id>"`` entries for per-specimen visibility.
        """
        if len(presence_counts) != self.num_animals:
            raise ValueError(
                f"presence_counts has {len(presence_counts)} entries, expected num_animals={self.num_animals}"
            )

        losses: List[torch.Tensor] = []
        counts: List[int] = []
        components: List[Dict[str, torch.Tensor]] = []
        ids: List[str] = []

        for index in range(self.num_animals):
            count = int(presence_counts[index])
            if self.drop_absent and count == 0:
                continue

            specimen_weights = weights_for_specimen(loss_weights, index, self.scene_loss_keys)
            result = loss_fn(index, specimen_weights)
            if result is None:
                continue
            loss, specimen_components = result
            if loss is None:
                continue

            losses.append(loss)
            counts.append(count)
            components.append(specimen_components or {})
            ids.append(self.specimen_ids[index])

        if not losses:
            zero = torch.zeros((), device=device, requires_grad=True)
            return (zero, {}) if return_components else zero

        total = aggregate_specimen_losses(losses, counts, reduction=self.reduction)

        if not return_components:
            return total

        merged = merge_loss_components(components, counts, specimen_ids=ids, reduction=self.reduction)
        merged["num_specimens_supervised"] = torch.tensor(float(len(losses)), device=total.device)
        return total, merged


def presence_counts_from_mask(animal_mask: torch.Tensor) -> List[int]:
    """Per-specimen presence counts from a ``(B, N)`` boolean mask."""
    if animal_mask.dim() != 2:
        raise ValueError(f"expected a (B, N) mask, got shape {tuple(animal_mask.shape)}")
    return [int(v) for v in animal_mask.sum(dim=0).tolist()]


def apply_visibility_floor(
    animal_mask: torch.Tensor,
    visible_keypoint_counts: torch.Tensor,
    min_visible: int,
) -> torch.Tensor:
    """Mark heavily occluded specimens absent for the samples in which they are.

    Design doc §10: once animals overlap, a specimen can be in frame but have
    almost nothing visible.  Supervising its 2D keypoints then teaches the model
    to hallucinate.  This turns "too few visible keypoints" into plain absence,
    which every downstream availability mask already understands.

    Args:
        animal_mask: ``(B, N)`` boolean presence mask.
        visible_keypoint_counts: ``(B, N)`` number of visible keypoints per
            specimen and sample (summed over views for multi-view).
        min_visible: Threshold; ``<= 0`` disables the floor.

    Returns:
        A new ``(B, N)`` mask.
    """
    if min_visible <= 0:
        return animal_mask
    if animal_mask.shape != visible_keypoint_counts.shape:
        raise ValueError(
            f"mask shape {tuple(animal_mask.shape)} != visibility shape {tuple(visible_keypoint_counts.shape)}"
        )
    return animal_mask & (visible_keypoint_counts >= min_visible)
