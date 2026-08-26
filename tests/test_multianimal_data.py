"""
Tests for the multi-animal data layer: collate, dataset adapters and the HDF5
layout declaration.

The behaviour that matters here is backwards compatibility (an existing
single-animal dataset must flow through unchanged as ``N = 1``) and rectangular
batches (a 2-mouse clip and a 3-mouse clip must be batchable without the
specimens shifting slots).
"""

import numpy as np
import pytest
import torch

from smal_fitter.neuralSMIL.multianimal.collate import (
    compose_multianimal_collate,
    make_multianimal_collate_fn,
    multianimal_collate_fn,
    pad_sample_to_num_animals,
)
from smal_fitter.neuralSMIL.multianimal.datasets import (
    GroupedSpecimenDataset,
    MultiAnimalDatasetAdapter,
    sleap_frame_key,
    sleap_track_specimen_key,
)
from smal_fitter.neuralSMIL.multianimal.hdf5_schema import (
    ANIMAL_MASK_DATASET,
    MultiAnimalHDF5Error,
    NUM_ANIMALS_ATTR,
    SPECIMEN_IDS_ATTR,
    VERSION_ATTR,
    animal_axis_shape,
    describe_layout,
    detect_layout,
    read_animal_mask,
    read_animal_slice,
    validate_file_shapes,
)
from smal_fitter.neuralSMIL.multianimal.schema import (
    ANIMAL_MASK_KEY,
    ANIMALS_KEY,
    NUM_ANIMALS_KEY,
    SPECIMEN_IDS_KEY,
    animal_mask_of,
    num_animals_of,
)


class LegacyDataset(torch.utils.data.Dataset):
    """A conventional single-animal dataset."""

    def __init__(self, length=4):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_data = {"input_image_data": np.full((2, 2, 3), index, dtype=np.float32)}
        y_data = {"root_rot": np.full(3, index, dtype=np.float32), "shape_betas": np.zeros(5, dtype=np.float32)}
        return x_data, y_data

    def get_target_resolution(self):
        return 224


class PerTrackDataset(torch.utils.data.Dataset):
    """One sample per (frame, track) pair, as multi-animal trackers store data."""

    def __init__(self, frames=3, tracks=("mouse_a", "mouse_b"), missing=()):
        self.entries = []
        for frame in range(frames):
            for track in tracks:
                if (frame, track) in missing:
                    continue
                self.entries.append((frame, track))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        frame, track = self.entries[index]
        x_data = {
            "input_image_data": np.full((2, 2, 3), frame, dtype=np.float32),
            "session_name": "session0",
            "frame_idx": frame,
            "track_id": track,
        }
        y_data = {
            "root_rot": np.full(3, hash(track) % 7, dtype=np.float32),
            "camera_intrinsics": np.eye(3, dtype=np.float32),
        }
        return x_data, y_data


class TestPadding:
    def test_legacy_sample_is_promoted_and_padded(self):
        x_data, y_data = LegacyDataset()[0]
        padded_x, padded_y = pad_sample_to_num_animals(x_data, y_data, 3)

        assert padded_x[NUM_ANIMALS_KEY] == 3
        assert list(padded_x[ANIMAL_MASK_KEY]) == [True, False, False]
        assert len(padded_y[ANIMALS_KEY]) == 3
        assert padded_y[ANIMALS_KEY][1]["root_rot"] is None

    def test_padding_preserves_slot_order(self):
        x_data, y_data = LegacyDataset()[0]
        padded_x, padded_y = pad_sample_to_num_animals(x_data, y_data, 3, specimen_ids=["a", "b", "c"])
        assert padded_x[SPECIMEN_IDS_KEY] == ["a", "b", "c"]
        assert padded_y[ANIMALS_KEY][0]["root_rot"] is not None

    def test_truncation_drops_trailing_slots(self):
        x_data, y_data = LegacyDataset()[0]
        wide_x, wide_y = pad_sample_to_num_animals(x_data, y_data, 4)
        narrow_x, narrow_y = pad_sample_to_num_animals(wide_x, wide_y, 2)
        assert num_animals_of(narrow_x, narrow_y) == 2

    def test_inputs_are_not_mutated(self):
        x_data, y_data = LegacyDataset()[0]
        pad_sample_to_num_animals(x_data, y_data, 3)
        assert ANIMALS_KEY not in y_data

    def test_rejects_zero_slots(self):
        x_data, y_data = LegacyDataset()[0]
        with pytest.raises(ValueError, match="num_animals must be >= 1"):
            pad_sample_to_num_animals(x_data, y_data, 0)


class TestCollate:
    def test_returns_two_lists(self):
        batch = [LegacyDataset()[i] for i in range(3)]
        x_batch, y_batch = multianimal_collate_fn(batch, num_animals=2)
        assert len(x_batch) == len(y_batch) == 3

    def test_mixed_group_sizes_become_rectangular(self):
        two = pad_sample_to_num_animals(*LegacyDataset()[0], 2)
        three = pad_sample_to_num_animals(*LegacyDataset()[1], 3)
        x_batch, y_batch = multianimal_collate_fn([two, three])
        assert {num_animals_of(x, y) for x, y in zip(x_batch, y_batch)} == {3}

    def test_explicit_n_wins_over_the_batch_maximum(self):
        three = pad_sample_to_num_animals(*LegacyDataset()[0], 3)
        x_batch, y_batch = multianimal_collate_fn([three], num_animals=2)
        assert num_animals_of(x_batch[0], y_batch[0]) == 2

    def test_identity_ordering_is_stamped_on_every_sample(self):
        batch = [LegacyDataset()[i] for i in range(3)]
        x_batch, _ = multianimal_collate_fn(batch, num_animals=2, specimen_ids=["m1", "m2"])
        assert all(x[SPECIMEN_IDS_KEY] == ["m1", "m2"] for x in x_batch)

    def test_empty_batch(self):
        assert multianimal_collate_fn([]) == ([], [])

    def test_factory_binds_the_slot_count(self):
        collate = make_multianimal_collate_fn(3, ["a", "b", "c"])
        x_batch, y_batch = collate([LegacyDataset()[0]])
        assert num_animals_of(x_batch[0], y_batch[0]) == 3
        assert "N3" in collate.__name__


class TestComposedCollate:
    def base_collate(self, batch):
        self.base_calls = getattr(self, "base_calls", 0) + 1
        return [x for x, _ in batch], [y for _, y in batch]

    def test_the_existing_collate_still_runs(self):
        collate = compose_multianimal_collate(self.base_collate, 2)
        collate([LegacyDataset()[0]])
        assert self.base_calls == 1

    def test_the_animal_axis_is_added_on_top(self):
        collate = compose_multianimal_collate(self.base_collate, 3, ["a", "b", "c"])
        x_batch, y_batch = collate([LegacyDataset()[0], LegacyDataset()[1]])
        assert all(num_animals_of(x, y) == 3 for x, y in zip(x_batch, y_batch))
        assert all(x[SPECIMEN_IDS_KEY] == ["a", "b", "c"] for x in x_batch)

    def test_already_multi_animal_batches_pass_through(self):
        wide = pad_sample_to_num_animals(*LegacyDataset()[0], 2)
        collate = compose_multianimal_collate(self.base_collate, 2)
        x_batch, y_batch = collate([wide])
        assert num_animals_of(x_batch[0], y_batch[0]) == 2

    def test_name_is_traceable(self):
        collate = compose_multianimal_collate(self.base_collate, 2)
        assert "N2" in collate.__name__


class TestFactory:
    def test_disabled_config_returns_everything_unchanged(self):
        from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
        from smal_fitter.neuralSMIL.multianimal import factory

        disabled = MultiAnimalConfig()
        dataset = LegacyDataset()
        sentinel = object()

        assert factory.wrap_dataset(dataset, disabled) is dataset
        assert factory.resolve_collate_fn(sentinel, disabled) is sentinel
        assert factory.build_regressor(
            disabled, single_animal_factory=lambda **kw: "single", multi_animal_factory=lambda **kw: "multi"
        ) == "single"
        assert "disabled" in factory.describe(disabled)

    def test_enabled_config_switches_every_piece(self):
        from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfig
        from smal_fitter.neuralSMIL.multianimal import factory

        enabled = MultiAnimalConfig(enabled=True, num_animals=2, specimen_ids=["m1", "m2"])
        dataset = factory.wrap_dataset(LegacyDataset(), enabled)
        x_data, y_data = dataset[0]

        assert num_animals_of(x_data, y_data) == 2
        assert factory.build_regressor(
            enabled,
            single_animal_factory=lambda **kw: "single",
            multi_animal_factory=lambda **kw: f"multi:{kw['multi_animal'].num_animals}",
        ) == "multi:2"
        assert "2 specimens" in factory.describe(enabled)

    def test_config_is_read_from_a_legacy_dict(self):
        from smal_fitter.neuralSMIL.multianimal import factory

        resolved = factory.resolve_multi_animal_config(
            {"multi_animal": {"enabled": True, "num_animals": 3}, "batch_size": 4}
        )
        assert resolved.num_animals == 3
        assert resolved.specimen_ids == ["specimen_0", "specimen_1", "specimen_2"]

    def test_missing_section_is_disabled(self):
        from smal_fitter.neuralSMIL.multianimal import factory

        assert factory.resolve_multi_animal_config({"batch_size": 4}).enabled is False

    def test_invalid_section_fails_at_startup(self):
        from smal_fitter.neuralSMIL.configs.multianimal_config import MultiAnimalConfigError
        from smal_fitter.neuralSMIL.multianimal import factory

        with pytest.raises(MultiAnimalConfigError):
            factory.resolve_multi_animal_config(
                {"multi_animal": {"enabled": True, "num_animals": 2, "head_strategy": "detr"}}
            )


class TestDatasetAdapter:
    def test_legacy_dataset_becomes_n_equals_one(self):
        adapter = MultiAnimalDatasetAdapter(LegacyDataset())
        x_data, y_data = adapter[0]
        assert num_animals_of(x_data, y_data) == 1
        assert list(animal_mask_of(x_data, y_data)) == [True]

    def test_length_is_preserved(self):
        assert len(MultiAnimalDatasetAdapter(LegacyDataset(7))) == 7

    def test_pads_to_the_configured_slot_count(self):
        adapter = MultiAnimalDatasetAdapter(LegacyDataset(), num_animals=3, specimen_ids=["a", "b", "c"])
        x_data, y_data = adapter[0]
        assert num_animals_of(x_data, y_data) == 3
        assert list(animal_mask_of(x_data, y_data)) == [True, False, False]

    def test_wrapped_dataset_attributes_are_forwarded(self):
        adapter = MultiAnimalDatasetAdapter(LegacyDataset())
        assert adapter.get_target_resolution() == 224

    def test_unknown_attribute_raises_attribute_error(self):
        adapter = MultiAnimalDatasetAdapter(LegacyDataset())
        with pytest.raises(AttributeError):
            _ = adapter.no_such_method

    def test_works_in_a_dataloader(self):
        loader = torch.utils.data.DataLoader(
            MultiAnimalDatasetAdapter(LegacyDataset(4), num_animals=2),
            batch_size=2,
            collate_fn=make_multianimal_collate_fn(2),
        )
        batches = list(loader)
        assert len(batches) == 2
        x_batch, y_batch = batches[0]
        assert len(x_batch) == 2
        assert num_animals_of(x_batch[0], y_batch[0]) == 2


class TestGroupedSpecimenDataset:
    def make(self, **kwargs):
        return GroupedSpecimenDataset(
            PerTrackDataset(**kwargs),
            frame_key_fn=sleap_frame_key,
            specimen_key_fn=sleap_track_specimen_key,
            specimen_order=["mouse_a", "mouse_b"],
            scene_keys=("camera_intrinsics",),
        )

    def test_one_sample_per_frame(self):
        grouped = self.make(frames=3)
        assert len(grouped) == 3

    def test_specimens_land_in_the_configured_slots(self):
        grouped = self.make(frames=2)
        x_data, y_data = grouped[0]
        assert x_data[SPECIMEN_IDS_KEY] == ["mouse_a", "mouse_b"]
        assert len(y_data[ANIMALS_KEY]) == 2

    def test_a_missing_track_becomes_an_absent_slot(self):
        grouped = self.make(frames=2, missing={(0, "mouse_b")})
        x_data, y_data = grouped[0]
        assert list(animal_mask_of(x_data, y_data)) == [True, False]
        assert y_data[ANIMALS_KEY][1]["root_rot"] is None

    def test_scene_keys_are_hoisted_out_of_the_specimen_dicts(self):
        grouped = self.make(frames=1)
        _, y_data = grouped[0]
        assert "camera_intrinsics" in y_data
        assert "camera_intrinsics" not in y_data[ANIMALS_KEY][0]

    def test_slot_order_is_stable_across_frames(self):
        # The invariant that removes the need for Hungarian matching.
        grouped = self.make(frames=3, missing={(1, "mouse_a")})
        orders = [grouped[i][0][SPECIMEN_IDS_KEY] for i in range(len(grouped))]
        assert all(order == ["mouse_a", "mouse_b"] for order in orders)

    def test_discovered_order_when_none_is_configured(self):
        grouped = GroupedSpecimenDataset(
            PerTrackDataset(frames=2),
            frame_key_fn=sleap_frame_key,
            specimen_key_fn=sleap_track_specimen_key,
        )
        assert grouped.specimen_ids == ["mouse_a", "mouse_b"]
        assert grouped.num_animals == 2

    def test_track_key_falls_back_to_zero(self):
        assert sleap_track_specimen_key({}, {}) == 0

    def test_frame_key_uses_session_and_frame(self):
        assert sleap_frame_key({"session_name": "s", "frame_idx": 4}, {}) == ("s", 4)


class FakeGroup(dict):
    """Minimal stand-in for an ``h5py`` group: dict access plus ``attrs``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {}


class FakeDataset:
    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, item):
        return self.array[item]


def make_fake_h5(num_animals=3, num_samples=2, declare=True, mask=None):
    keypoints = FakeDataset(np.zeros((num_samples, num_animals, 4, 6, 2), dtype=np.float32))
    global_rot = FakeDataset(np.arange(num_samples * num_animals * 3, dtype=np.float32).reshape(
        num_samples, num_animals, 3
    ))
    animal_mask = FakeDataset(
        np.ones((num_samples, num_animals), dtype=bool) if mask is None else np.asarray(mask)
    )

    root = FakeGroup(
        {
            "metadata": FakeGroup(),
            "multiview_keypoints": FakeGroup({"keypoints_2d": keypoints}),
            "parameters": FakeGroup({"global_rot": global_rot}),
            "auxiliary": FakeGroup({"animal_mask": animal_mask}),
            "multiview_images": FakeGroup(
                {"images": FakeDataset(np.zeros((num_samples, 4, 10), dtype=np.uint8))}
            ),
        }
    )
    if declare:
        root["metadata"].attrs[VERSION_ATTR] = 1
        root["metadata"].attrs[NUM_ANIMALS_ATTR] = num_animals
        root["metadata"].attrs[SPECIMEN_IDS_ATTR] = np.array(
            [f"mouse_{i}".encode() for i in range(num_animals)], dtype=object
        )
    return root


class TestHDF5Layout:
    def test_undeclared_file_reads_as_single_animal(self):
        layout = detect_layout(make_fake_h5(declare=False))
        assert layout.multi_animal is False
        assert layout.num_animals == 1
        assert layout.specimen_ids == ["specimen_0"]

    def test_declared_file_reports_ids(self):
        layout = detect_layout(make_fake_h5(num_animals=3))
        assert layout.multi_animal is True
        assert layout.num_animals == 3
        assert layout.specimen_ids == ["mouse_0", "mouse_1", "mouse_2"]

    def test_future_schema_version_is_rejected(self):
        h5 = make_fake_h5()
        h5["metadata"].attrs[VERSION_ATTR] = 99
        with pytest.raises(MultiAnimalHDF5Error, match="upgrade SMILify"):
            detect_layout(h5)

    def test_id_count_mismatch_is_rejected(self):
        h5 = make_fake_h5(num_animals=3)
        h5["metadata"].attrs[SPECIMEN_IDS_ATTR] = np.array([b"only_one"], dtype=object)
        with pytest.raises(MultiAnimalHDF5Error, match=SPECIMEN_IDS_ATTR):
            detect_layout(h5)

    def test_shape_validation_passes_on_a_well_formed_file(self):
        validate_file_shapes(make_fake_h5(num_animals=3))

    def test_missing_animal_axis_is_reported(self):
        h5 = make_fake_h5(num_animals=3)
        h5["parameters"]["global_rot"] = FakeDataset(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(MultiAnimalHDF5Error, match="parameters/global_rot"):
            validate_file_shapes(h5)

    def test_missing_animal_mask_is_reported(self):
        h5 = make_fake_h5(num_animals=3)
        del h5["auxiliary"]["animal_mask"]
        with pytest.raises(MultiAnimalHDF5Error, match=ANIMAL_MASK_DATASET):
            validate_file_shapes(h5)

    def test_single_animal_file_skips_validation(self):
        validate_file_shapes(make_fake_h5(declare=False))

    def test_animal_axis_shape_insertion(self):
        assert animal_axis_shape((100, 4, 55, 2), 3) == (100, 3, 4, 55, 2)

    def test_animal_axis_shape_rejects_scalar(self):
        with pytest.raises(MultiAnimalHDF5Error, match="scalar"):
            animal_axis_shape((), 3)

    def test_reading_a_specimen_slice(self):
        h5 = make_fake_h5(num_animals=3, num_samples=2)
        value = read_animal_slice(h5, "parameters/global_rot", 1, 2)
        assert value.tolist() == pytest.approx([15.0, 16.0, 17.0])

    def test_reading_an_absent_dataset_returns_none(self):
        assert read_animal_slice(make_fake_h5(), "parameters/does_not_exist", 0, 0) is None

    def test_reading_the_mask(self):
        h5 = make_fake_h5(num_animals=2, num_samples=2, mask=[[True, False], [True, True]])
        assert read_animal_mask(h5, 0, 2).tolist() == [True, False]

    def test_mask_defaults_to_all_present_when_absent(self):
        h5 = make_fake_h5(num_animals=3)
        del h5["auxiliary"]["animal_mask"]
        assert read_animal_mask(h5, 0, 3).tolist() == [True, True, True]

    def test_describe_layout_is_serialisable(self):
        described = describe_layout(detect_layout(make_fake_h5(num_animals=2)))
        assert described == {
            "multi_animal": True,
            "num_animals": 2,
            "specimen_ids": ["mouse_0", "mouse_1"],
            "schema_version": 1,
        }
