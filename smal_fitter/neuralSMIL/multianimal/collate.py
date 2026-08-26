"""
Collate functions for multi-animal batches.

The multi-animal contract keeps a sample as a plain ``(x_data, y_data)`` pair of
dicts (see :mod:`.schema`), so the existing "list of dicts" collate style still
applies and no tensor stacking happens here.  What *does* need doing at collate
time is making the batch rectangular along the animal axis: a clip with two mice
and a clip with three must agree on ``N`` before the shared heads can be indexed
by specimen.  Padding is by absence, which every downstream availability mask
already understands.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import (
    ANIMAL_MASK_KEY,
    ANIMALS_KEY,
    NUM_ANIMALS_KEY,
    SPECIMEN_IDS_KEY,
    absent_specimen_targets,
    animal_mask_of,
    is_multi_animal,
    num_animals_of,
    specimen_ids_of,
    wrap_single_animal,
)

Sample = Tuple[Dict[str, Any], Dict[str, Any]]


def pad_sample_to_num_animals(
    x_data: Dict[str, Any],
    y_data: Dict[str, Any],
    num_animals: int,
    specimen_ids: Optional[Sequence[str]] = None,
) -> Sample:
    """Grow (or truncate) a sample to exactly ``num_animals`` specimen slots.

    Added slots are marked absent and filled with all-``None`` targets, so they
    contribute nothing to the loss.  Truncation drops trailing slots and is only
    correct when the caller has already fixed the identity ordering — it exists
    so a run configured for ``N = 2`` can consume a 3-mouse clip's first two
    tracks rather than crashing.

    Args:
        x_data: Scene-level inputs.
        y_data: Targets, multi-animal or legacy single-animal.
        num_animals: Target slot count.
        specimen_ids: Optional canonical id list to stamp on the sample.  Using
            the run's configured ids here is what keeps the ordering stable
            across a dataset that does not always list every specimen.

    Returns:
        A new ``(x_data, y_data)`` pair; the inputs are not mutated.
    """
    if num_animals < 1:
        raise ValueError(f"num_animals must be >= 1, got {num_animals}")

    if not is_multi_animal(x_data, y_data):
        x_data, y_data = wrap_single_animal(x_data, y_data)

    current = num_animals_of(x_data, y_data)
    mask = animal_mask_of(x_data, y_data, current)
    animals = list(y_data.get(ANIMALS_KEY, []))
    ids = specimen_ids_of(x_data, y_data, current)

    new_animals: List[Dict[str, Any]] = []
    new_mask = np.zeros(num_animals, dtype=bool)
    new_ids: List[str] = []

    for index in range(num_animals):
        if index < current and index < len(animals) and bool(mask[index]):
            new_animals.append(animals[index])
            new_mask[index] = True
        else:
            new_animals.append(absent_specimen_targets())
        if specimen_ids is not None and index < len(specimen_ids):
            new_ids.append(str(specimen_ids[index]))
        elif index < len(ids):
            new_ids.append(ids[index])
        else:
            new_ids.append(f"specimen_{index}")

    new_x = dict(x_data)
    new_x[NUM_ANIMALS_KEY] = num_animals
    new_x[ANIMAL_MASK_KEY] = new_mask
    new_x[SPECIMEN_IDS_KEY] = new_ids

    new_y = dict(y_data)
    new_y[ANIMALS_KEY] = new_animals
    return new_x, new_y


def multianimal_collate_fn(
    batch: Sequence[Sample],
    num_animals: Optional[int] = None,
    specimen_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Collate multi-animal samples into ``(x_data_batch, y_data_batch)``.

    Mirrors the existing single-animal / multi-view collate contract: nothing is
    stacked, the model does its own batching, and the only work done here is
    making the animal axis rectangular.

    Args:
        batch: Sequence of ``(x_data, y_data)`` samples.
        num_animals: Slot count to pad to.  Defaults to the largest ``N`` in the
            batch, which is the right behaviour for a dataset of mixed group
            sizes; pass the run's configured ``N`` to keep it fixed across
            batches (recommended, since head *i* is bound to specimen *i*).
        specimen_ids: Canonical identity ordering to stamp on every sample.

    Returns:
        ``(x_data_batch, y_data_batch)`` — two lists of dicts.
    """
    if not batch:
        return [], []

    target_n = int(num_animals) if num_animals is not None else max(num_animals_of(x, y) for x, y in batch)

    x_batch: List[Dict[str, Any]] = []
    y_batch: List[Dict[str, Any]] = []
    for x_data, y_data in batch:
        padded_x, padded_y = pad_sample_to_num_animals(x_data, y_data, target_n, specimen_ids=specimen_ids)
        x_batch.append(padded_x)
        y_batch.append(padded_y)
    return x_batch, y_batch


def make_multianimal_collate_fn(num_animals: int, specimen_ids: Optional[Sequence[str]] = None):
    """Bind ``num_animals`` / ``specimen_ids`` into a picklable-friendly collate.

    ``DataLoader(..., collate_fn=make_multianimal_collate_fn(3, ids))`` keeps the
    slot count fixed for every batch of a run, which is what strict head ↔
    specimen correspondence requires.
    """
    ids = list(specimen_ids) if specimen_ids is not None else None

    def _collate(batch: Sequence[Sample]):
        return multianimal_collate_fn(batch, num_animals=num_animals, specimen_ids=ids)

    _collate.__name__ = f"multianimal_collate_fn_N{num_animals}"
    return _collate


def compose_multianimal_collate(
    base_collate: Callable[[Sequence[Sample]], Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]],
    num_animals: int,
    specimen_ids: Optional[Sequence[str]] = None,
):
    """Add the animal axis on top of an existing collate function.

    The training scripts' own collate functions do more than unzip the batch —
    they log dataset composition, carry ``available_labels``, and so on — so the
    multi-animal step *wraps* them rather than replacing them: the existing
    collate runs first and its output is then padded to ``num_animals`` slots.

    Args:
        base_collate: The single-animal collate, returning
            ``(x_data_batch, y_data_batch)``.
        num_animals: Slot count to pad to.
        specimen_ids: Canonical identity ordering stamped on every sample.

    Returns:
        A collate function with the same call signature as ``base_collate``.
    """
    ids = list(specimen_ids) if specimen_ids is not None else None

    def _collate(batch: Sequence[Sample]):
        x_batch, y_batch = base_collate(batch)
        padded = [
            pad_sample_to_num_animals(x_data, y_data, num_animals, specimen_ids=ids)
            for x_data, y_data in zip(x_batch, y_batch)
        ]
        return [x for x, _ in padded], [y for _, y in padded]

    _collate.__name__ = f"multianimal_{getattr(base_collate, '__name__', 'collate')}_N{num_animals}"
    return _collate
