"""
Tests for the multi-animal data contract (``multianimal/schema.py``).

These cover the invariants the rest of the multi-animal stack assumes: legacy
samples promote to ``N = 1`` without copying, specimen slicing produces ordinary
single-animal target dicts, absent specimens neutralise into all-``None``
targets, and a shuffled specimen ordering is rejected rather than silently
training head 0 on two different animals.
"""

import numpy as np
import pytest

from smal_fitter.neuralSMIL.multianimal.collate import pad_sample_to_num_animals
from smal_fitter.neuralSMIL.multianimal.schema import (
    ANIMAL_MASK_KEY,
    ANIMALS_KEY,
    NUM_ANIMALS_KEY,
    SPECIMEN_IDS_KEY,
    MultiAnimalSchemaError,
    absent_specimen_targets,
    animal_mask_of,
    assert_stable_identity,
    is_multi_animal,
    make_multi_animal_sample,
    num_animals_of,
    specimen_ids_of,
    specimen_input_view,
    specimen_target_view,
    split_batch_by_specimen,
    validate_sample,
    wrap_single_animal,
)


def single_animal_sample(seed: int = 0):
    """A legacy single-animal ``(x_data, y_data)`` pair."""
    rng = np.random.default_rng(seed)
    x_data = {
        "input_image_data": rng.random((8, 8, 3), dtype=np.float32),
        "dataset_source": "unit_test",
    }
    y_data = {
        "root_rot": rng.random(3).astype(np.float32),
        "root_loc": rng.random(3).astype(np.float32),
        "joint_angles": rng.random((4, 3)).astype(np.float32),
        "shape_betas": rng.random(5).astype(np.float32),
        "keypoints_2d": rng.random((6, 2)).astype(np.float32),
        "keypoint_visibility": np.ones(6, dtype=bool),
    }
    return x_data, y_data


def multi_animal_sample(num_animals=3, present=(True, True, False), seed=0):
    """A multi-animal sample with the given presence pattern."""
    x_data, _ = single_animal_sample(seed)
    scene = {"cam_fov_per_view": np.array([[8.0]], dtype=np.float32)}
    animals = []
    for index in range(num_animals):
        if present[index]:
            _, y_data = single_animal_sample(seed + index + 1)
            animals.append(y_data)
        else:
            animals.append(None)
    return make_multi_animal_sample(
        x_data, scene, animals, specimen_ids=[f"mouse_{i}" for i in range(num_animals)]
    )


class TestDetection:
    def test_legacy_sample_is_not_multi_animal(self):
        x_data, y_data = single_animal_sample()
        assert not is_multi_animal(x_data, y_data)
        assert num_animals_of(x_data, y_data) == 1

    def test_wrapped_sample_is_multi_animal(self):
        x_data, y_data = wrap_single_animal(*single_animal_sample())
        assert is_multi_animal(x_data, y_data)
        assert num_animals_of(x_data, y_data) == 1
        assert x_data[NUM_ANIMALS_KEY] == 1
        assert list(x_data[ANIMAL_MASK_KEY]) == [True]

    def test_wrap_does_not_copy_the_targets(self):
        x_data, y_data = single_animal_sample()
        _, wrapped_y = wrap_single_animal(x_data, y_data)
        # The per-specimen entry is the same object: N=1 promotion is free.
        assert wrapped_y[ANIMALS_KEY][0] is y_data

    def test_wrap_is_idempotent(self):
        sample = wrap_single_animal(*single_animal_sample())
        again = wrap_single_animal(*sample)
        assert again[0] is sample[0]
        assert again[1] is sample[1]

    def test_default_specimen_ids(self):
        x_data, y_data = wrap_single_animal(*single_animal_sample())
        del x_data[SPECIMEN_IDS_KEY]
        assert specimen_ids_of(x_data, y_data, 3) == ["specimen_0", "specimen_1", "specimen_2"]


class TestAnimalMask:
    def test_mask_reflects_absence(self):
        x_data, y_data = multi_animal_sample(3, present=(True, False, True))
        assert list(animal_mask_of(x_data, y_data)) == [True, False, True]

    def test_mask_pads_with_absent_slots(self):
        x_data, y_data = multi_animal_sample(2, present=(True, True))
        assert list(animal_mask_of(x_data, y_data, num_animals=4)) == [True, True, False, False]

    def test_mask_truncates(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, True))
        assert list(animal_mask_of(x_data, y_data, num_animals=2)) == [True, True]

    def test_legacy_sample_defaults_to_present(self):
        x_data, y_data = single_animal_sample()
        assert list(animal_mask_of(x_data, y_data)) == [True]


class TestSpecimenViews:
    def test_target_view_is_a_plain_single_animal_dict(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, False))
        view = specimen_target_view(y_data, 1, 3)
        assert ANIMALS_KEY not in view
        assert view["root_rot"] is y_data[ANIMALS_KEY][1]["root_rot"]

    def test_scene_keys_are_shared_by_every_specimen(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, True))
        first = specimen_target_view(y_data, 0, 3)
        second = specimen_target_view(y_data, 1, 3)
        assert first["cam_fov_per_view"] is second["cam_fov_per_view"]

    def test_absent_specimen_targets_are_all_none(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, False))
        view = specimen_target_view(y_data, 2, 3, present=False)
        for key in ("root_rot", "joint_angles", "shape_betas"):
            assert view[key] is None
        assert view["has_3d_data"] is False

    def test_absent_specimen_omits_the_keypoint_keys(self):
        # Present-but-None keypoints would make assemble_batch_inputs build a
        # keypoint_data entry that _validate_sample_visibility then indexes.
        x_data, y_data = multi_animal_sample(3, present=(True, True, False))
        view = specimen_target_view(y_data, 2, 3, present=False)
        for key in ("keypoints_2d", "keypoint_visibility", "keypoints_3d"):
            assert key not in view

    def test_a_promoted_legacy_sample_does_not_leak_into_other_slots(self):
        # wrap_single_animal keeps the labels at both the top level and slot 0;
        # slot 1 must still come out empty rather than inheriting slot 0's.
        x_data, y_data = wrap_single_animal(*single_animal_sample())
        x_data, y_data = pad_sample_to_num_animals(x_data, y_data, 2)
        view = specimen_target_view(y_data, 1, 2, present=False)
        assert view["root_rot"] is None
        assert "keypoints_2d" not in view

    def test_absent_targets_helper_covers_the_boolean_flags(self):
        targets = absent_specimen_targets()
        assert targets["has_ground_truth_betas"] is False
        assert targets["has_ground_truth_pose"] is False

    def test_out_of_range_specimen_raises(self):
        _, y_data = multi_animal_sample(2, present=(True, True))
        with pytest.raises(MultiAnimalSchemaError, match="out of range"):
            specimen_target_view(y_data, 5, 2)

    def test_input_view_keeps_scene_inputs_and_adds_identity(self):
        x_data, _ = multi_animal_sample(3, present=(True, True, True))
        view = specimen_input_view(x_data, 1, 3)
        assert view["input_image_data"] is x_data["input_image_data"]
        assert view["specimen_index"] == 1
        assert view["specimen_id"] == "mouse_1"
        assert ANIMAL_MASK_KEY not in view

    def test_input_view_masks_labels_of_an_absent_specimen(self):
        x_data, _ = multi_animal_sample(3, present=(True, True, False))
        x_data["available_labels"] = {"joint_rot": True, "betas": True}
        view = specimen_input_view(x_data, 2, 3, present=False)
        assert view["available_labels"] == {"joint_rot": False, "betas": False}

    def test_input_view_does_not_fabricate_available_labels(self):
        # Regression: an absent specimen used to gain an empty available_labels
        # dict, which made the key present for some batch rows and absent for
        # others. The downstream merge then read the short list as "nothing is
        # available" and silently deleted the PRESENT specimens' supervision.
        x_data, _ = multi_animal_sample(3, present=(True, True, False))
        assert "available_labels" not in x_data
        view = specimen_input_view(x_data, 2, 3, present=False)
        assert "available_labels" not in view

    def test_input_view_supports_per_specimen_available_labels(self):
        x_data, _ = multi_animal_sample(2, present=(True, True))
        x_data["available_labels"] = [{"betas": True}, {"betas": False}]
        assert specimen_input_view(x_data, 0, 2)["available_labels"] == {"betas": True}
        assert specimen_input_view(x_data, 1, 2)["available_labels"] == {"betas": False}


class TestBatchSplit:
    def test_split_keeps_batch_shape_and_reports_presence(self):
        batch = [multi_animal_sample(3, present=p, seed=i) for i, p in enumerate([(True, True, False), (True, False, True)])]
        x_batch = [x for x, _ in batch]
        y_batch = [y for _, y in batch]

        x_slice, y_slice, present = split_batch_by_specimen(x_batch, y_batch, 2, 3)

        assert len(x_slice) == len(y_slice) == 2  # batch shape preserved
        assert list(present) == [False, True]
        assert y_slice[0]["root_rot"] is None  # absent -> neutralised
        assert y_slice[1]["root_rot"] is not None


class TestValidation:
    def test_valid_sample_passes(self):
        validate_sample(*multi_animal_sample(3, present=(True, True, True)))

    def test_legacy_sample_is_rejected_with_a_helpful_message(self):
        x_data, y_data = single_animal_sample()
        with pytest.raises(MultiAnimalSchemaError, match="wrap_single_animal"):
            validate_sample(x_data, y_data)

    def test_mask_length_mismatch_is_reported(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, True))
        x_data[ANIMAL_MASK_KEY] = np.ones(2, dtype=bool)
        with pytest.raises(MultiAnimalSchemaError, match=ANIMAL_MASK_KEY):
            validate_sample(x_data, y_data)

    def test_declared_count_mismatch_is_reported(self):
        x_data, y_data = multi_animal_sample(3, present=(True, True, True))
        x_data[NUM_ANIMALS_KEY] = 2
        with pytest.raises(MultiAnimalSchemaError, match="disagrees"):
            validate_sample(x_data, y_data)

    def test_expected_count_mismatch_is_reported(self):
        sample = multi_animal_sample(3, present=(True, True, True))
        with pytest.raises(MultiAnimalSchemaError, match="expected 2"):
            validate_sample(*sample, expected_num_animals=2)

    def test_non_dict_specimen_entry_is_reported(self):
        x_data, y_data = multi_animal_sample(2, present=(True, True))
        y_data[ANIMALS_KEY][1] = "not a dict"
        with pytest.raises(MultiAnimalSchemaError, match="must be a dict"):
            validate_sample(x_data, y_data)


class TestStableIdentity:
    def test_consistent_ordering_passes(self):
        assert_stable_identity([["a", "b"], ["a", "b"], ["a", "b"]])

    def test_empty_batch_passes(self):
        assert_stable_identity([])

    def test_shuffled_ordering_is_rejected(self):
        # This is the failure mode that would silently train head 0 on two
        # different animals, so it must be loud.
        with pytest.raises(MultiAnimalSchemaError, match="not stable"):
            assert_stable_identity([["a", "b"], ["b", "a"]])
