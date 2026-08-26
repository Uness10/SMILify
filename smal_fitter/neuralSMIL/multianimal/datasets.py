"""
Dataset adapters for the animal axis.

Two problems are solved here, both of them about *not* rewriting the existing
datasets:

:class:`MultiAnimalDatasetAdapter`
    Wraps any dataset that yields ``(x_data, y_data)`` and presents it as a
    multi-animal dataset.  A legacy single-animal dataset becomes an ``N = 1``
    multi-animal dataset with no copying and no preprocessing, which is what
    makes existing HDF5 files and checkpoints keep working (and what the
    equivalence test exercises).

:class:`GroupedSpecimenDataset`
    Builds a multi-animal dataset out of *per-specimen* annotations that share a
    frame.  This is the shape multi-animal tracking data actually arrives in —
    SLEAP stores one instance per track per frame — so rather than teaching every
    loader about groups, a loader keeps returning per-specimen samples and this
    adapter groups them by frame key in a fixed track order.

Both preserve the invariant the whole design rests on: specimen ``i`` is always
the same animal, in every sample, for the whole run.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import torch

from .collate import pad_sample_to_num_animals
from .schema import (
    ANIMALS_KEY,
    is_multi_animal,
    make_multi_animal_sample,
    num_animals_of,
    validate_sample,
    wrap_single_animal,
)

Sample = Tuple[Dict[str, Any], Dict[str, Any]]


class MultiAnimalDatasetAdapter(torch.utils.data.Dataset):
    """Present any ``(x_data, y_data)`` dataset as a multi-animal dataset.

    Args:
        dataset: The wrapped dataset.  May already be multi-animal.
        num_animals: Slot count to expose.  Samples with fewer specimens are
            padded with absent slots; with more, the trailing slots are dropped.
            Defaults to the wrapped dataset's own count (1 for legacy datasets).
        specimen_ids: Canonical identity ordering stamped on every sample.
        validate: Check each produced sample against the schema.  Cheap, and it
            turns a silently mis-shaped dataset into a clear error at the first
            batch rather than a confusing loss curve.

    Every attribute the wrapped dataset exposes (``get_target_resolution``,
    ``get_canonical_camera_order``, ...) is forwarded, so the adapter is a
    drop-in replacement wherever a dataset object is passed around.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_animals: Optional[int] = None,
        specimen_ids: Optional[Sequence[str]] = None,
        validate: bool = True,
    ):
        self.dataset = dataset
        self.num_animals = int(num_animals) if num_animals is not None else None
        self.specimen_ids = list(specimen_ids) if specimen_ids is not None else None
        self.validate = bool(validate)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        x_data, y_data = self.dataset[index]

        if not is_multi_animal(x_data, y_data):
            x_data, y_data = wrap_single_animal(x_data, y_data)

        target_n = self.num_animals if self.num_animals is not None else num_animals_of(x_data, y_data)
        x_data, y_data = pad_sample_to_num_animals(x_data, y_data, target_n, specimen_ids=self.specimen_ids)

        if self.validate:
            validate_sample(x_data, y_data, expected_num_animals=target_n)
        return x_data, y_data

    def __getattr__(self, name: str):
        # Only reached for attributes this instance does not have, so it cannot
        # shadow `dataset`, `num_animals`, ... set in __init__.  Guarded so that
        # an access before __init__ (e.g. during unpickling) raises
        # AttributeError instead of recursing.
        return _forward_attr(self, name)


class GroupedSpecimenDataset(torch.utils.data.Dataset):
    """Group per-specimen samples that share a frame into multi-animal samples.

    Multi-animal tracking data is naturally stored one instance per track per
    frame.  This adapter reads such a dataset once, buckets its samples by a
    caller-supplied frame key, and emits one multi-animal sample per frame with
    the specimens placed in a **fixed slot order** derived from their identity
    (track) label.

    Args:
        dataset: Dataset yielding per-specimen ``(x_data, y_data)`` samples.
        frame_key_fn: ``(x_data, y_data) -> Hashable`` identifying the frame (or
            frame group, for multi-view) a sample belongs to.
        specimen_key_fn: ``(x_data, y_data) -> Hashable`` identifying *which*
            animal the sample is.  This is the identity that pins a specimen to
            a head; for SLEAP it is the track id.
        specimen_order: Explicit slot order.  Strongly recommended: it makes the
            head ↔ specimen binding a property of the *configuration* rather
            than of whatever order the file happened to be written in.  When
            omitted, first-seen order is used and recorded.
        scene_keys: ``y_data`` keys that describe the frame rather than the
            animal (cameras, image sizes, ...) and are therefore taken from the
            first specimen of the group.
        eager_index: Build the frame index at construction time.  Requires one
            pass over the dataset; set False if that pass is expensive and the
            caller supplies ``index`` instead.
        index: Precomputed ``{frame_key: {specimen_key: dataset_index}}``.

    Note:
        The wrapped dataset's ``__getitem__`` is called once per specimen, so
        for image data this reads the same frame ``N`` times.  That is fine for
        HDF5-backed loaders (the decode is cached by the OS page cache) but for
        an expensive loader prefer storing the animal axis natively — see
        :mod:`.hdf5_schema`.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        frame_key_fn: Callable[[Dict[str, Any], Dict[str, Any]], Hashable],
        specimen_key_fn: Callable[[Dict[str, Any], Dict[str, Any]], Hashable],
        specimen_order: Optional[Sequence[Hashable]] = None,
        scene_keys: Sequence[str] = (),
        eager_index: bool = True,
        index: Optional["OrderedDict[Hashable, Dict[Hashable, int]]"] = None,
    ):
        self.dataset = dataset
        self.frame_key_fn = frame_key_fn
        self.specimen_key_fn = specimen_key_fn
        self.scene_keys = tuple(scene_keys)
        self.specimen_order: List[Hashable] = list(specimen_order) if specimen_order is not None else []
        self._index: "OrderedDict[Hashable, Dict[Hashable, int]]" = index if index is not None else OrderedDict()
        self._frame_keys: List[Hashable] = list(self._index.keys())

        if index is None and eager_index:
            self.build_index()

    def build_index(self) -> None:
        """Scan the wrapped dataset and bucket its samples by frame."""
        self._index = OrderedDict()
        discovered: List[Hashable] = []

        for i in range(len(self.dataset)):
            x_data, y_data = self.dataset[i]
            frame_key = self.frame_key_fn(x_data, y_data)
            specimen_key = self.specimen_key_fn(x_data, y_data)
            self._index.setdefault(frame_key, {})[specimen_key] = i
            if specimen_key not in discovered:
                discovered.append(specimen_key)

        if not self.specimen_order:
            self.specimen_order = discovered
        self._frame_keys = list(self._index.keys())

    @property
    def num_animals(self) -> int:
        """Number of specimen slots, i.e. the length of the identity ordering."""
        return len(self.specimen_order)

    @property
    def specimen_ids(self) -> List[str]:
        """Slot identity labels, in slot order."""
        return [str(key) for key in self.specimen_order]

    def __len__(self) -> int:
        return len(self._frame_keys)

    def __getitem__(self, index: int) -> Sample:
        frame_key = self._frame_keys[index]
        members = self._index[frame_key]

        scene_x: Optional[Dict[str, Any]] = None
        scene_y: Dict[str, Any] = {}
        animal_targets: List[Optional[Dict[str, Any]]] = []

        for specimen_key in self.specimen_order:
            dataset_index = members.get(specimen_key)
            if dataset_index is None:
                animal_targets.append(None)
                continue

            x_data, y_data = self.dataset[dataset_index]
            if scene_x is None:
                scene_x = dict(x_data)
                scene_y = {key: y_data[key] for key in self.scene_keys if key in y_data}
            animal_targets.append({k: v for k, v in y_data.items() if k not in self.scene_keys and k != ANIMALS_KEY})

        if scene_x is None:
            raise IndexError(f"frame group {frame_key!r} has no present specimen")

        return make_multi_animal_sample(scene_x, scene_y, animal_targets, specimen_ids=self.specimen_ids)

    def __getattr__(self, name: str):
        return _forward_attr(self, name)


def _forward_attr(wrapper: Any, name: str):
    """Forward an unknown attribute to the wrapped dataset.

    Raises ``AttributeError`` (never ``KeyError``) when the wrapper has no
    wrapped dataset yet, so ``hasattr`` and unpickling behave normally.
    """
    inner = wrapper.__dict__.get("dataset")
    if inner is None:
        raise AttributeError(f"{type(wrapper).__name__!r} object has no attribute {name!r}")
    return getattr(inner, name)


def sleap_track_specimen_key(x_data: Dict[str, Any], y_data: Dict[str, Any]) -> Hashable:
    """Identity key for SLEAP-style data: the track id, else the instance index.

    SLEAP's ``tracks`` array carries an explicit identity axis, which is exactly
    the "fixed identity/order established beforehand" the design requires — no
    identity discovery, no matching.
    """
    del y_data
    for key in ("track_id", "track", "specimen_id", "instance_id"):
        if key in x_data and x_data[key] is not None:
            return x_data[key]
    return 0


def sleap_frame_key(x_data: Dict[str, Any], y_data: Dict[str, Any]) -> Hashable:
    """Frame key for SLEAP-style data: ``(session, frame index)``."""
    del y_data
    session = x_data.get("session_name", x_data.get("session", ""))
    frame = x_data.get("frame_idx", x_data.get("frame_index", x_data.get("frame", 0)))
    return (session, int(frame))
