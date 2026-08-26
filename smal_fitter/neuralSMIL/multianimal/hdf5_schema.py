"""
On-disk (HDF5) layout for multi-animal datasets.

The preprocessed multi-view layout written by
``replicAntMultiViewPreprocessor`` / the SLEAP preprocessors stores one animal
per sample::

    multiview_keypoints/keypoints_2d          (S, V, J, 2)
    multiview_keypoints/keypoint_visibility   (S, V, J)
    parameters/global_rot                     (S, 3)
    parameters/joint_rot                      (S, N_POSE, 3)
    ...

The multi-animal layout inserts an **animal axis immediately after the sample
axis** on every animal-level dataset, and leaves scene-level datasets (images,
cameras, view masks) exactly as they are::

    multiview_keypoints/keypoints_2d          (S, A, V, J, 2)
    multiview_keypoints/keypoint_visibility   (S, A, V, J)
    parameters/global_rot                     (S, A, 3)
    parameters/joint_rot                      (S, A, N_POSE, 3)
    auxiliary/animal_mask                     (S, A)          bool
    multiview_images/...                      unchanged
    multiview_keypoints/camera_*              unchanged

plus file-level attributes::

    metadata.attrs["multianimal_schema_version"] = 1
    metadata.attrs["num_animals"]               = A
    metadata.attrs["specimen_ids"]              = ["mouse_a", "mouse_b", ...]

Why an axis rather than N separate sample rows: the images and cameras are
shared by the specimens of a frame, so duplicating a row per animal would
multiply the (dominant) image storage by ``A`` and make it possible for the
specimens of one frame to drift apart under shuffling or filtering.

A file *without* ``multianimal_schema_version`` is a single-animal file and is
read as ``A = 1`` — that is the whole backwards-compatibility story, and
:func:`detect_layout` is the one place that decides it.

``h5py`` is imported lazily so this module (and therefore the schema constants)
can be imported anywhere without the dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import SCHEMA_VERSION

#: Attribute names on the ``metadata`` group.
VERSION_ATTR = "multianimal_schema_version"
NUM_ANIMALS_ATTR = "num_animals"
SPECIMEN_IDS_ATTR = "specimen_ids"

#: Dataset path of the per-sample, per-specimen presence mask.
ANIMAL_MASK_DATASET = "auxiliary/animal_mask"

#: Datasets that gain the animal axis, mapped to the rank they must have *with*
#: it.  The rank matters because shape alone is ambiguous: a ``(S, 3)``
#: ``global_rot`` in a 3-specimen file looks exactly like ``(S, A)`` if only the
#: second dimension is checked, so a file that was never migrated would pass a
#: shape-only test.  ``None`` means "rank varies with configuration" (per-joint
#: vs PCA scale/translation blocks), where only the axis size is checked.
ANIMAL_AXIS_DATASET_RANKS: Dict[str, Optional[int]] = {
    # (S, A, V, J, 2) / (S, A, V, J)
    "multiview_keypoints/keypoints_2d": 5,
    "multiview_keypoints/keypoint_visibility": 4,
    "multiview_keypoints/keypoints_3d": None,
    # single-view variants: (S, A, J, 2) / (S, A, J)
    "keypoints/keypoints_2d": 4,
    "keypoints/keypoint_visibility": 3,
    "keypoints/keypoints_3d": None,
    # (S, A, 3) / (S, A, N_POSE, 3) / (S, A, N_BETAS)
    "parameters/global_rot": 3,
    "parameters/joint_rot": 4,
    "parameters/betas": 3,
    "parameters/trans": 3,
    "parameters/log_beta_scales": None,
    "parameters/betas_trans": None,
    # (S, A)
    "auxiliary/has_3d_data": 2,
    "auxiliary/has_gt_betas": 2,
    "auxiliary/has_gt_pose": 2,
}

#: Paths of the datasets that gain the animal axis.
ANIMAL_AXIS_DATASETS: Tuple[str, ...] = tuple(ANIMAL_AXIS_DATASET_RANKS)

#: Datasets that stay scene level and must NOT gain the animal axis.
SCENE_AXIS_DATASETS: Tuple[str, ...] = (
    "multiview_images/images",
    "multiview_images/view_mask",
    "multiview_keypoints/camera_indices",
    "multiview_keypoints/camera_intrinsics",
    "multiview_keypoints/camera_extrinsics_R",
    "multiview_keypoints/camera_extrinsics_t",
    "multiview_keypoints/image_sizes",
    "auxiliary/num_views",
    "auxiliary/frame_idx",
    "auxiliary/canonical_to_world_R",
    "auxiliary/canonical_to_world_t",
    "auxiliary/canonical_cam_id",
    "auxiliary/session_name",
    "auxiliary/camera_names",
)


class MultiAnimalHDF5Error(ValueError):
    """Raised when a file does not satisfy the multi-animal HDF5 layout."""


@dataclass(frozen=True)
class DatasetLayout:
    """What :func:`detect_layout` found in a file.

    Attributes:
        multi_animal: True when the file declares the animal axis.
        num_animals: ``A`` (1 for a legacy single-animal file).
        specimen_ids: Identity label per slot, in slot order.
        schema_version: Declared schema version (0 for legacy files).
    """

    multi_animal: bool
    num_animals: int
    specimen_ids: List[str] = field(default_factory=list)
    schema_version: int = 0

    def __post_init__(self):
        if not self.specimen_ids:
            object.__setattr__(self, "specimen_ids", [f"specimen_{i}" for i in range(self.num_animals)])


def detect_layout(h5_file: Any) -> DatasetLayout:
    """Read the multi-animal declaration off an open HDF5 file.

    Args:
        h5_file: An open ``h5py.File`` (or any mapping-like object exposing
            ``attrs`` on a ``metadata`` member).

    Returns:
        A :class:`DatasetLayout`.  Files with no declaration are reported as
        single-animal, which is what makes every existing preprocessed dataset
        load through the multi-animal path unchanged.
    """
    metadata = h5_file["metadata"] if "metadata" in h5_file else None
    attrs = getattr(metadata, "attrs", {}) if metadata is not None else {}

    if VERSION_ATTR not in attrs:
        return DatasetLayout(multi_animal=False, num_animals=1, schema_version=0)

    version = int(attrs[VERSION_ATTR])
    if version > SCHEMA_VERSION:
        raise MultiAnimalHDF5Error(
            f"file declares multianimal schema version {version} but this code understands "
            f"at most {SCHEMA_VERSION}; upgrade SMILify or re-export the dataset."
        )

    num_animals = int(attrs.get(NUM_ANIMALS_ATTR, 1))
    if num_animals < 1:
        raise MultiAnimalHDF5Error(f"'{NUM_ANIMALS_ATTR}' must be >= 1, got {num_animals}")

    ids = [_as_str(value) for value in np.atleast_1d(attrs.get(SPECIMEN_IDS_ATTR, []))]
    if ids and len(ids) != num_animals:
        raise MultiAnimalHDF5Error(
            f"'{SPECIMEN_IDS_ATTR}' has {len(ids)} entries but '{NUM_ANIMALS_ATTR}' is {num_animals}"
        )

    return DatasetLayout(
        multi_animal=True,
        num_animals=num_animals,
        specimen_ids=ids,
        schema_version=version,
    )


def declare_layout(
    h5_file: Any,
    num_animals: int,
    specimen_ids: Optional[Sequence[str]] = None,
) -> None:
    """Stamp the multi-animal declaration onto a file being written.

    Call this from a preprocessing script after creating the ``metadata`` group.
    A file that carries the animal axis on its datasets but *not* this
    declaration would be silently misread as single-animal, so writers must
    always call it.
    """
    if num_animals < 1:
        raise MultiAnimalHDF5Error(f"num_animals must be >= 1, got {num_animals}")

    metadata = h5_file.require_group("metadata")
    metadata.attrs[VERSION_ATTR] = SCHEMA_VERSION
    metadata.attrs[NUM_ANIMALS_ATTR] = int(num_animals)

    ids = list(specimen_ids) if specimen_ids else [f"specimen_{i}" for i in range(num_animals)]
    if len(ids) != num_animals:
        raise MultiAnimalHDF5Error(f"specimen_ids has {len(ids)} entries, expected num_animals={num_animals}")
    metadata.attrs[SPECIMEN_IDS_ATTR] = np.array([str(value) for value in ids], dtype=object)


def animal_axis_shape(single_animal_shape: Sequence[int], num_animals: int) -> Tuple[int, ...]:
    """Insert the animal axis into a single-animal dataset shape.

    ``(S, V, J, 2)`` with ``A = 3`` becomes ``(S, 3, V, J, 2)``.  Use this in a
    preprocessing script so the axis always lands in the documented position.
    """
    shape = tuple(int(dim) for dim in single_animal_shape)
    if not shape:
        raise MultiAnimalHDF5Error("cannot insert an animal axis into a scalar shape")
    if num_animals < 1:
        raise MultiAnimalHDF5Error(f"num_animals must be >= 1, got {num_animals}")
    return (shape[0], int(num_animals)) + shape[1:]


def validate_file_shapes(h5_file: Any, layout: Optional[DatasetLayout] = None) -> None:
    """Check that a multi-animal file's datasets really carry the animal axis.

    Verifies that every present :data:`ANIMAL_AXIS_DATASETS` entry has ``A`` as
    its second dimension and that no :data:`SCENE_AXIS_DATASETS` entry does, so
    a half-migrated file fails loudly at open time rather than producing subtly
    misaligned specimens.

    Raises:
        MultiAnimalHDF5Error: naming the first dataset that disagrees.
    """
    layout = layout or detect_layout(h5_file)
    if not layout.multi_animal:
        return

    num_samples = None
    for path, expected_rank in ANIMAL_AXIS_DATASET_RANKS.items():
        dataset = _maybe_get(h5_file, path)
        if dataset is None:
            continue
        shape = tuple(dataset.shape)
        if len(shape) < 2 or shape[1] != layout.num_animals:
            raise MultiAnimalHDF5Error(
                f"dataset '{path}' has shape {shape}; expected the animal axis of size "
                f"{layout.num_animals} at position 1"
            )
        if expected_rank is not None and len(shape) != expected_rank:
            raise MultiAnimalHDF5Error(
                f"dataset '{path}' has shape {shape} (rank {len(shape)}) but a multi-animal "
                f"file must store it with rank {expected_rank}. The animal axis is most likely "
                "missing and the second dimension coincidentally equals "
                f"num_animals={layout.num_animals}."
            )
        num_samples = shape[0] if num_samples is None else num_samples

    mask = _maybe_get(h5_file, ANIMAL_MASK_DATASET)
    if mask is None:
        raise MultiAnimalHDF5Error(
            f"multi-animal file is missing '{ANIMAL_MASK_DATASET}'; without it an absent "
            "specimen cannot be distinguished from one whose labels are simply zero."
        )
    if tuple(mask.shape)[1:] != (layout.num_animals,):
        raise MultiAnimalHDF5Error(
            f"'{ANIMAL_MASK_DATASET}' has shape {tuple(mask.shape)}, expected (S, {layout.num_animals})"
        )

    for path in SCENE_AXIS_DATASETS:
        dataset = _maybe_get(h5_file, path)
        if dataset is None or num_samples is None:
            continue
        shape = tuple(dataset.shape)
        if len(shape) >= 2 and shape[0] == num_samples and shape[1] == layout.num_animals:
            # Ambiguous only when A happens to equal the real second dimension;
            # flagged rather than guessed, because silently treating a camera
            # array as per-animal would corrupt every projection.
            raise MultiAnimalHDF5Error(
                f"scene-level dataset '{path}' has shape {shape}, whose second axis matches "
                f"num_animals={layout.num_animals}. Scene-level data must not carry the animal "
                "axis; check the preprocessing script."
            )


def read_animal_slice(h5_file: Any, path: str, sample_index: int, specimen_index: int) -> Optional[np.ndarray]:
    """Read one specimen's slice of an animal-axis dataset.

    Returns ``None`` when the dataset is absent, so a reader can express
    "this label does not exist in this file" without branching on the layout.
    """
    dataset = _maybe_get(h5_file, path)
    if dataset is None:
        return None
    return np.asarray(dataset[sample_index, specimen_index])


def read_animal_mask(h5_file: Any, sample_index: int, num_animals: int) -> np.ndarray:
    """Read the ``(A,)`` presence mask for one sample, defaulting to all-present."""
    dataset = _maybe_get(h5_file, ANIMAL_MASK_DATASET)
    if dataset is None:
        return np.ones(num_animals, dtype=bool)
    return np.asarray(dataset[sample_index], dtype=bool).reshape(-1)[:num_animals]


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _maybe_get(h5_file: Any, path: str):
    """``h5_file[path]`` or ``None`` when any component of the path is missing."""
    node = h5_file
    for part in path.split("/"):
        try:
            if part not in node:
                return None
        except TypeError:
            return None
        node = node[part]
    return node


def _as_str(value: Any) -> str:
    """Decode an HDF5 string attribute, which may be ``bytes``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def describe_layout(layout: DatasetLayout) -> Dict[str, Any]:
    """Serialisable summary, for dataset-info printouts and run manifests."""
    return {
        "multi_animal": layout.multi_animal,
        "num_animals": layout.num_animals,
        "specimen_ids": list(layout.specimen_ids),
        "schema_version": layout.schema_version,
    }
