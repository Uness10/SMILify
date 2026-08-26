"""
Multi-animal sample schema (the "animal axis" data contract).

This module is the single source of truth for how a sample containing *N known
specimens* is represented in memory, and it is deliberately dependency-free
(numpy only) so that datasets, collate functions, trainers and tests can all
agree on the contract without importing torch, pytorch3d or ``config``.

Design (see ``docs/design/multianimal.md``)
------------------------------------------
A sample is ``V`` images (``V == 1`` for single-view) containing ``N`` specimens
with a **fixed identity/order**::

    sample = V images containing N known specimens
    specimen_i: 2D keypoints, pose, shape, translation

Two levels of information are therefore distinguished:

``scene level``
    Everything that belongs to the *image(s)*, not to any one animal: the
    images themselves, the camera(s), view masks, dataset provenance.  These
    keys keep their existing names at the top level of ``x_data`` / ``y_data``
    so that every existing consumer keeps working.

``animal level``
    Pose, shape, translation and 2D/3D keypoints — one set per specimen.  These
    live under :data:`ANIMALS_KEY` as a *list of per-specimen dicts*, each dict
    using exactly the same key names a single-animal sample already uses.

Why a list of dicts rather than a leading ``(N, ...)`` axis on every array?

* Per-specimen dicts are byte-for-byte what the existing single-animal target
  collectors (``_collect_body_targets_batch``, ``_extract_target_parameters_single``)
  already consume, so the whole loss stack is reused unchanged.
* Labels are frequently *ragged*: specimen 1 may have pose GT while specimen 2
  only has 2D keypoints.  ``None`` per key per specimen expresses that directly.
* Backwards compatibility is a one-line promotion (:func:`wrap_single_animal`).

On-disk (HDF5) storage uses a dense leading animal axis instead — see
:mod:`smal_fitter.neuralSMIL.multianimal.hdf5_schema` — and is converted to this
in-memory form by the dataset's ``__getitem__``.

Identity
--------
Head ``i`` is permanently bound to specimen ``i``.  There is no Hungarian
matching and no permutation-invariant loss; the *dataset* is responsible for
emitting specimens in a stable order (e.g. a SLEAP track id).  :data:`SPECIMEN_IDS_KEY`
carries the identity labels so that this ordering can be asserted end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Key names
# --------------------------------------------------------------------------- #

#: ``y_data`` key holding the list of per-specimen target dicts.
ANIMALS_KEY = "animals"

#: ``x_data`` key holding a ``(N,)`` boolean array: True where specimen *i* is
#: actually present in this sample.  Mirrors the existing ``view_mask``.
ANIMAL_MASK_KEY = "animal_mask"

#: ``x_data`` key holding the number of specimen slots ``N`` in this sample.
NUM_ANIMALS_KEY = "num_animals"

#: ``x_data`` key holding the stable identity label of each specimen slot.
SPECIMEN_IDS_KEY = "specimen_ids"

#: Schema version written into HDF5 files and checkpoints.
SCHEMA_VERSION = 1


#: Target keys that belong to a single specimen.  Anything not listed here is
#: treated as scene level and is shared by every specimen.
ANIMAL_LEVEL_TARGET_KEYS: Tuple[str, ...] = (
    "root_rot",
    "root_loc",
    "joint_angles",
    "shape_betas",
    "log_beta_scales",
    "betas_trans",
    "keypoints_2d",
    "keypoint_visibility",
    "keypoints_3d",
    "has_3d_data",
    "has_ground_truth_betas",
    "has_ground_truth_pose",
    "mesh_scale",
    # Per-joint scale/translation PCA weights consumed by
    # SMILImageRegressor._extract_target_parameters_single in
    # scale_trans_mode='separate'/'entangled_with_betas'.
    "scale_weights",
    "trans_weights",
    "translation_factor",
)

#: Animal-level keys that must be **omitted** for an absent specimen rather than
#: set to ``None``.  Existing consumers branch on key *presence* — e.g.
#: ``SMILImageRegressor.assemble_batch_inputs`` builds a ``keypoint_data`` entry
#: whenever ``keypoints_2d`` and ``keypoint_visibility`` are both present, and
#: ``_validate_sample_visibility`` then indexes them unconditionally.  Leaving
#: them present-but-``None`` would raise instead of reading as "no keypoints".
PRESENCE_SENSITIVE_TARGET_KEYS: Tuple[str, ...] = (
    "keypoints_2d",
    "keypoint_visibility",
    "keypoints_3d",
)

#: Loss components that are properties of the *scene*, not of an animal.  They
#: must be supervised exactly once per sample even though the per-specimen loss
#: is evaluated ``N`` times.  See
#: :class:`~smal_fitter.neuralSMIL.multianimal.losses.MultiAnimalLossAggregator`.
SCENE_LEVEL_LOSS_KEYS: Tuple[str, ...] = ("fov", "cam_rot", "cam_trans")

#: Prediction keys that are properties of the scene rather than of an animal.
SCENE_LEVEL_PREDICTION_KEYS: Tuple[str, ...] = (
    "fov",
    "cam_rot",
    "cam_trans",
    "fov_per_view",
    "cam_rot_per_view",
    "cam_trans_per_view",
    "num_views",
    "view_mask",
    "camera_indices",
)


class MultiAnimalSchemaError(ValueError):
    """Raised when a sample does not satisfy the multi-animal contract."""


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #


def is_multi_animal(x_data: Optional[Dict[str, Any]], y_data: Optional[Dict[str, Any]]) -> bool:
    """Return True when the sample already carries an explicit animal axis.

    A sample counts as multi-animal as soon as it declares one, even if it
    declares ``N == 1``: that distinguishes "this dataset knows about specimens"
    from "this is a legacy single-animal sample".
    """
    if y_data is not None and ANIMALS_KEY in y_data:
        return True
    if x_data is not None and (NUM_ANIMALS_KEY in x_data or ANIMAL_MASK_KEY in x_data):
        return True
    return False


def num_animals_of(x_data: Optional[Dict[str, Any]], y_data: Optional[Dict[str, Any]]) -> int:
    """Number of specimen slots ``N`` declared by a sample (1 for legacy samples)."""
    if y_data is not None and ANIMALS_KEY in y_data:
        return len(y_data[ANIMALS_KEY])
    if x_data is not None and NUM_ANIMALS_KEY in x_data:
        return int(x_data[NUM_ANIMALS_KEY])
    if x_data is not None and ANIMAL_MASK_KEY in x_data:
        return int(len(np.asarray(x_data[ANIMAL_MASK_KEY]).reshape(-1)))
    return 1


def animal_mask_of(
    x_data: Optional[Dict[str, Any]],
    y_data: Optional[Dict[str, Any]],
    num_animals: Optional[int] = None,
) -> np.ndarray:
    """Return the ``(N,)`` boolean presence mask for a sample.

    Falls back to "all present" for legacy samples that carry no mask.  When
    ``num_animals`` is larger than the declared slot count the mask is padded
    with ``False`` — that is how a batch mixing 2-mouse and 3-mouse clips is
    padded up to a common ``N``.
    """
    declared = num_animals_of(x_data, y_data)
    n = int(num_animals) if num_animals is not None else declared

    if x_data is not None and ANIMAL_MASK_KEY in x_data:
        mask = np.asarray(x_data[ANIMAL_MASK_KEY], dtype=bool).reshape(-1)
    else:
        mask = np.ones(declared, dtype=bool)

    if mask.shape[0] == n:
        return mask
    if mask.shape[0] > n:
        return mask[:n]
    padded = np.zeros(n, dtype=bool)
    padded[: mask.shape[0]] = mask
    return padded


def specimen_ids_of(
    x_data: Optional[Dict[str, Any]],
    y_data: Optional[Dict[str, Any]],
    num_animals: Optional[int] = None,
) -> List[str]:
    """Return the stable identity label of each specimen slot.

    Defaults to ``["specimen_0", "specimen_1", ...]`` so that downstream code
    (logging, per-specimen metrics, visualisation) always has a name to use.
    """
    n = int(num_animals) if num_animals is not None else num_animals_of(x_data, y_data)
    ids: Sequence[Any]
    if x_data is not None and SPECIMEN_IDS_KEY in x_data:
        ids = list(x_data[SPECIMEN_IDS_KEY])
    else:
        ids = []
    out = [str(ids[i]) if i < len(ids) else f"specimen_{i}" for i in range(n)]
    return out


# --------------------------------------------------------------------------- #
# Construction / promotion
# --------------------------------------------------------------------------- #


def wrap_single_animal(
    x_data: Dict[str, Any],
    y_data: Dict[str, Any],
    specimen_id: str = "specimen_0",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Promote a legacy single-animal ``(x_data, y_data)`` pair to ``N == 1``.

    The returned dicts are shallow copies; the per-specimen target dict is the
    *same object* as ``y_data`` so no data is duplicated.  Every existing
    single-animal dataset therefore flows through the multi-animal stack with
    zero preprocessing and provably identical numerics (see
    ``tests/test_multianimal_equivalence.py``).
    """
    if is_multi_animal(x_data, y_data):
        return x_data, y_data

    new_x = dict(x_data)
    new_x[NUM_ANIMALS_KEY] = 1
    new_x[ANIMAL_MASK_KEY] = np.ones(1, dtype=bool)
    new_x[SPECIMEN_IDS_KEY] = [specimen_id]

    new_y = dict(y_data)
    new_y[ANIMALS_KEY] = [y_data]
    return new_x, new_y


def make_multi_animal_sample(
    x_data: Dict[str, Any],
    scene_targets: Dict[str, Any],
    animal_targets: Sequence[Optional[Dict[str, Any]]],
    specimen_ids: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Assemble a multi-animal sample from scene-level and per-specimen parts.

    Args:
        x_data: Scene-level inputs (images, camera indices, view mask, ...).
        scene_targets: Scene-level targets (camera parameters, image sizes, ...).
        animal_targets: One dict per specimen slot, in **fixed identity order**.
            ``None`` marks an absent specimen; its mask entry becomes ``False``
            and it is materialised as an all-``None`` target dict so that every
            existing availability mask switches off for it.
        specimen_ids: Optional stable identity labels, one per slot.

    Returns:
        ``(x_data, y_data)`` satisfying the multi-animal contract.
    """
    n = len(animal_targets)
    mask = np.array([t is not None for t in animal_targets], dtype=bool)

    new_x = dict(x_data)
    new_x[NUM_ANIMALS_KEY] = n
    new_x[ANIMAL_MASK_KEY] = mask
    new_x[SPECIMEN_IDS_KEY] = (
        [str(s) for s in specimen_ids] if specimen_ids is not None else [f"specimen_{i}" for i in range(n)]
    )

    new_y = dict(scene_targets)
    new_y[ANIMALS_KEY] = [t if t is not None else absent_specimen_targets() for t in animal_targets]
    return new_x, new_y


def absent_specimen_targets() -> Dict[str, Any]:
    """Target dict for a specimen slot that is not present in this sample.

    Every animal-level label is ``None`` so the existing implicit availability
    masking ("None detection", see ``SMILImageRegressor.predict_from_batch``)
    switches the whole per-specimen loss off, and every boolean availability
    flag is ``False``.  Keypoint keys are omitted entirely rather than set to
    ``None`` (see :data:`PRESENCE_SENSITIVE_TARGET_KEYS`).  Nothing else in the
    loss stack needs to know about the animal axis.
    """
    targets: Dict[str, Any] = {
        key: None for key in ANIMAL_LEVEL_TARGET_KEYS if key not in PRESENCE_SENSITIVE_TARGET_KEYS
    }
    targets["has_3d_data"] = False
    targets["has_ground_truth_betas"] = False
    targets["has_ground_truth_pose"] = False
    return targets


# --------------------------------------------------------------------------- #
# Slicing
# --------------------------------------------------------------------------- #


def specimen_target_view(
    y_data: Dict[str, Any],
    specimen_index: int,
    num_animals: Optional[int] = None,
    present: bool = True,
) -> Dict[str, Any]:
    """Project a multi-animal ``y_data`` onto one specimen.

    The result is an ordinary *single-animal* target dict: scene-level keys are
    carried through unchanged (the camera is shared) and the animal-level keys
    come from slot ``specimen_index``.  This is the adapter that lets the whole
    existing single-animal loss stack be reused per specimen.

    Args:
        y_data: Multi-animal (or legacy single-animal) target dict.
        specimen_index: Slot to project onto.
        num_animals: Declared slot count; inferred when omitted.
        present: When ``False`` the animal-level keys are forced to ``None``
            (used to switch off an absent specimen without touching the loss).
    """
    n = int(num_animals) if num_animals is not None else num_animals_of(None, y_data)
    if not 0 <= specimen_index < n:
        raise MultiAnimalSchemaError(f"specimen_index {specimen_index} out of range for num_animals={n}")

    animals = y_data.get(ANIMALS_KEY)
    if animals is None:
        # Legacy single-animal dict: the animal-level keys are already at the
        # top level, so the scene projection is the sample itself.
        return {k: v for k, v in y_data.items() if k != ANIMALS_KEY}

    # Drop every animal-level key from the scene projection before merging the
    # specimen's own labels. `wrap_single_animal` keeps the promoted sample's
    # labels at BOTH the top level and in slot 0, so without this a later
    # (or absent) slot would silently inherit specimen 0's keypoints and pose.
    scene = {
        k: v
        for k, v in y_data.items()
        if k != ANIMALS_KEY and k not in ANIMAL_LEVEL_TARGET_KEYS and k not in PRESENCE_SENSITIVE_TARGET_KEYS
    }

    if not present or specimen_index >= len(animals):
        scene.update(absent_specimen_targets())
        return scene

    scene.update(animals[specimen_index])
    return scene


def specimen_input_view(
    x_data: Dict[str, Any],
    specimen_index: int,
    num_animals: Optional[int] = None,
    present: bool = True,
) -> Dict[str, Any]:
    """Project a multi-animal ``x_data`` onto one specimen.

    Images and cameras are scene level and are shared verbatim.  The only
    per-specimen input is ``available_labels`` (multi-dataset label masking),
    which may itself be given per specimen as a list; an absent specimen gets an
    all-``False`` label map so it can never contribute to the loss.
    """
    n = int(num_animals) if num_animals is not None else num_animals_of(x_data, None)
    if not 0 <= specimen_index < n:
        raise MultiAnimalSchemaError(f"specimen_index {specimen_index} out of range for num_animals={n}")

    out = {k: v for k, v in x_data.items() if k not in (ANIMAL_MASK_KEY, NUM_ANIMALS_KEY, SPECIMEN_IDS_KEY)}

    labels = x_data.get("available_labels")
    if isinstance(labels, (list, tuple)):
        out["available_labels"] = dict(labels[specimen_index]) if specimen_index < len(labels) else {}
    elif isinstance(labels, dict):
        out["available_labels"] = dict(labels)

    # An absent specimen has every label switched off -- but only when the
    # sample declared labels in the first place. Fabricating an entry here
    # would make `available_labels` present for SOME rows of a batch and absent
    # for others, and the downstream merge treats a short list as "nothing is
    # available", silently deleting the *present* specimens' supervision too.
    # Absence is already fully expressed by the all-None targets.
    if not present and "available_labels" in out:
        out["available_labels"] = {key: False for key in out["available_labels"]}

    out["specimen_index"] = specimen_index
    out["specimen_id"] = specimen_ids_of(x_data, None, n)[specimen_index]
    return out


def split_batch_by_specimen(
    x_data_batch: Sequence[Dict[str, Any]],
    y_data_batch: Sequence[Dict[str, Any]],
    specimen_index: int,
    num_animals: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray]:
    """Project a whole batch onto one specimen slot.

    Returns ``(x_batch, y_batch, present_mask)`` where ``present_mask`` is a
    ``(B,)`` boolean array marking the batch entries in which this specimen is
    actually present.  Absent entries are *kept* (so the batch keeps its shape
    and the shared backbone/camera stay aligned) but are neutralised into
    all-``None`` targets, which the existing availability masking already
    handles.
    """
    x_out: List[Dict[str, Any]] = []
    y_out: List[Dict[str, Any]] = []
    present = np.zeros(len(x_data_batch), dtype=bool)

    for i, (x_data, y_data) in enumerate(zip(x_data_batch, y_data_batch)):
        mask = animal_mask_of(x_data, y_data, num_animals)
        is_present = bool(mask[specimen_index])
        present[i] = is_present
        x_out.append(specimen_input_view(x_data, specimen_index, num_animals, present=is_present))
        y_out.append(specimen_target_view(y_data, specimen_index, num_animals, present=is_present))

    return x_out, y_out, present


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_sample(
    x_data: Dict[str, Any],
    y_data: Dict[str, Any],
    expected_num_animals: Optional[int] = None,
) -> None:
    """Fail fast on a malformed multi-animal sample.

    Checks the invariants the rest of the stack relies on: a declared slot
    count that agrees across ``x_data`` and ``y_data``, a mask and an id list of
    that length, and per-specimen entries that are dicts.  Raises
    :class:`MultiAnimalSchemaError` with a message naming the offending key.
    """
    if not is_multi_animal(x_data, y_data):
        raise MultiAnimalSchemaError(
            f"sample carries no animal axis: expected '{ANIMALS_KEY}' in y_data or "
            f"'{NUM_ANIMALS_KEY}'/'{ANIMAL_MASK_KEY}' in x_data. Use wrap_single_animal() "
            "to promote a legacy single-animal sample."
        )

    n = num_animals_of(x_data, y_data)
    if n < 1:
        raise MultiAnimalSchemaError(f"num_animals must be >= 1, got {n}")
    if expected_num_animals is not None and n != expected_num_animals:
        raise MultiAnimalSchemaError(f"sample declares num_animals={n}, expected {expected_num_animals}")

    if NUM_ANIMALS_KEY in x_data and int(x_data[NUM_ANIMALS_KEY]) != n:
        raise MultiAnimalSchemaError(
            f"x_data['{NUM_ANIMALS_KEY}']={x_data[NUM_ANIMALS_KEY]} disagrees with "
            f"len(y_data['{ANIMALS_KEY}'])={n}"
        )

    if ANIMAL_MASK_KEY in x_data:
        mask = np.asarray(x_data[ANIMAL_MASK_KEY]).reshape(-1)
        if mask.shape[0] != n:
            raise MultiAnimalSchemaError(f"x_data['{ANIMAL_MASK_KEY}'] has length {mask.shape[0]}, expected {n}")

    if SPECIMEN_IDS_KEY in x_data and len(list(x_data[SPECIMEN_IDS_KEY])) != n:
        raise MultiAnimalSchemaError(
            f"x_data['{SPECIMEN_IDS_KEY}'] has length {len(list(x_data[SPECIMEN_IDS_KEY]))}, expected {n}"
        )

    animals = y_data.get(ANIMALS_KEY)
    if animals is not None:
        for i, entry in enumerate(animals):
            if not isinstance(entry, dict):
                raise MultiAnimalSchemaError(f"y_data['{ANIMALS_KEY}'][{i}] must be a dict, got {type(entry).__name__}")


def assert_stable_identity(batch_specimen_ids: Sequence[Sequence[str]]) -> None:
    """Assert that every sample in a batch uses the same specimen ordering.

    Strict head-to-specimen correspondence is what removes the need for
    Hungarian matching (design doc §5).  It only holds if the *dataset* emits a
    consistent ordering, so this check is cheap insurance against a silently
    reshuffled track order.
    """
    if not batch_specimen_ids:
        return
    reference = list(batch_specimen_ids[0])
    for i, ids in enumerate(batch_specimen_ids[1:], start=1):
        if list(ids) != reference:
            raise MultiAnimalSchemaError(
                "specimen ordering is not stable across the batch — head/specimen "
                f"correspondence would be broken. Sample 0 has {reference}, sample {i} has {list(ids)}."
            )
