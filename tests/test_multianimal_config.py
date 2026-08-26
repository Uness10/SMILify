"""
Tests for ``MultiAnimalConfig`` and its integration with the unified config system.

The multi-animal section has to survive the same path every other section does —
dataclass defaults, JSON deep-merge, validation, legacy-dict conversion — and it
has to be *off* by default so that no existing run changes behaviour.
"""

import json
import os
import tempfile

import pytest

from smal_fitter.neuralSMIL.configs import (
    MultiAnimalConfig,
    MultiAnimalConfigError,
    MultiViewConfig,
    SingleViewConfig,
    load_config,
)

EXAMPLES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "smal_fitter", "neuralSMIL", "configs", "examples"
)


class TestDefaults:
    def test_disabled_by_default(self):
        assert MultiAnimalConfig().enabled is False
        assert MultiAnimalConfig().is_active is False

    def test_present_on_both_modes_and_off(self):
        assert SingleViewConfig().multi_animal.enabled is False
        assert MultiViewConfig().multi_animal.enabled is False

    def test_default_specimen_ids(self):
        assert MultiAnimalConfig(num_animals=3).specimen_ids == ["specimen_0", "specimen_1", "specimen_2"]

    def test_disabled_config_skips_validation_of_the_rest(self):
        # A half-filled but disabled section must never block a single-animal run.
        MultiAnimalConfig(enabled=False, num_animals=0, head_strategy="nonsense").validate()


class TestNormalization:
    def test_auto_ids_are_regenerated_after_a_merge(self):
        # __post_init__ runs before the JSON merge, so num_animals arrives late.
        config = MultiAnimalConfig()
        config.num_animals = 3
        config.normalize()
        assert config.specimen_ids == ["specimen_0", "specimen_1", "specimen_2"]

    def test_authored_ids_are_not_overwritten(self):
        config = MultiAnimalConfig(num_animals=2, specimen_ids=["mouse_a", "mouse_b"])
        config.num_animals = 3
        config.normalize()
        assert config.specimen_ids == ["mouse_a", "mouse_b"]  # left alone -> validate() will complain

    def test_mismatch_after_normalize_is_reported(self):
        config = MultiAnimalConfig(enabled=True, num_animals=2, specimen_ids=["mouse_a", "mouse_b"])
        config.num_animals = 3
        with pytest.raises(MultiAnimalConfigError, match="specimen_ids has 2 entries"):
            config.validate()


class TestValidation:
    def test_valid_config_passes(self):
        MultiAnimalConfig(enabled=True, num_animals=3).validate()

    def test_zero_animals_is_rejected(self):
        with pytest.raises(MultiAnimalConfigError, match="num_animals must be >= 1"):
            MultiAnimalConfig(enabled=True, num_animals=0).validate()

    def test_unknown_head_strategy_is_rejected(self):
        with pytest.raises(MultiAnimalConfigError, match="head_strategy"):
            MultiAnimalConfig(enabled=True, num_animals=2, head_strategy="detr").validate()

    def test_unknown_loss_reduction_is_rejected(self):
        with pytest.raises(MultiAnimalConfigError, match="loss_reduction"):
            MultiAnimalConfig(enabled=True, num_animals=2, loss_reduction="median").validate()

    def test_duplicate_specimen_ids_are_rejected(self):
        with pytest.raises(MultiAnimalConfigError, match="must be unique"):
            MultiAnimalConfig(enabled=True, num_animals=2, specimen_ids=["m", "m"]).validate()

    def test_first_specimen_camera_is_rejected_for_several_animals(self):
        # Camera is scene level (design doc §6): routing it through one animal's
        # head would give the other heads no camera gradient at all.
        with pytest.raises(MultiAnimalConfigError, match="camera_mode='first_specimen'"):
            MultiAnimalConfig(enabled=True, num_animals=2, camera_mode="first_specimen").validate()

    def test_first_specimen_camera_is_allowed_for_one_animal(self):
        MultiAnimalConfig(enabled=True, num_animals=1, camera_mode="first_specimen").validate()

    def test_lr_scale_length_must_match(self):
        with pytest.raises(MultiAnimalConfigError, match="per_specimen_lr_scale"):
            MultiAnimalConfig(enabled=True, num_animals=3, per_specimen_lr_scale=[1.0, 1.0]).validate()

    def test_lr_scale_must_be_positive(self):
        with pytest.raises(MultiAnimalConfigError, match="must be > 0"):
            MultiAnimalConfig(enabled=True, num_animals=2, per_specimen_lr_scale=[1.0, 0.0]).validate()

    def test_negative_visibility_floor_is_rejected(self):
        with pytest.raises(MultiAnimalConfigError, match="min_visible_keypoints_per_specimen"):
            MultiAnimalConfig(enabled=True, num_animals=2, min_visible_keypoints_per_specimen=-1).validate()


class TestSerialization:
    def test_dict_round_trip(self):
        original = MultiAnimalConfig(
            enabled=True,
            num_animals=3,
            specimen_ids=["a", "b", "c"],
            head_strategy="shared_query",
            loss_reduction="weighted_mean",
            per_specimen_lr_scale=[1.0, 2.0, 0.5],
        )
        restored = MultiAnimalConfig.from_dict(original.to_dict())
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_ignores_unknown_keys(self):
        config = MultiAnimalConfig.from_dict({"enabled": True, "num_animals": 2, "not_a_field": 1})
        assert config.enabled is True
        assert config.num_animals == 2

    def test_from_dict_of_none_gives_defaults(self):
        assert MultiAnimalConfig.from_dict(None).enabled is False

    def test_appears_in_the_singleview_legacy_dict(self):
        config = SingleViewConfig()
        # Pin the resolution so the legacy conversion does not need to import the
        # backbone factory (and therefore timm) just to answer a config question.
        config.model.input_resolution = 224
        config.multi_animal.enabled = True
        config.multi_animal.num_animals = 2
        legacy = config.to_legacy_dict()
        assert legacy["multi_animal"]["enabled"] is True
        assert legacy["multi_animal"]["num_animals"] == 2
        assert legacy["multi_animal"]["specimen_ids"] == ["specimen_0", "specimen_1"]

    def test_appears_in_the_multiview_legacy_dict(self):
        config = MultiViewConfig()
        config.model.input_resolution = 224
        config.multi_animal.enabled = True
        config.multi_animal.num_animals = 3
        legacy = config.to_multiview_legacy_dict()
        assert legacy["multi_animal"]["num_animals"] == 3


class TestJSONLoading:
    def load(self, payload):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name
        try:
            return load_config(config_file=path)
        finally:
            os.unlink(path)

    def test_json_block_is_merged(self):
        config = self.load(
            {
                "mode": "multiview",
                "multi_animal": {
                    "enabled": True,
                    "num_animals": 3,
                    "specimen_ids": ["mouse_a", "mouse_b", "mouse_c"],
                    "head_strategy": "shared_query",
                },
            }
        )
        assert config.multi_animal.enabled is True
        assert config.multi_animal.num_animals == 3
        assert config.multi_animal.specimen_ids == ["mouse_a", "mouse_b", "mouse_c"]
        assert config.multi_animal.head_strategy == "shared_query"

    def test_ids_default_to_the_declared_count(self):
        config = self.load({"mode": "singleview", "multi_animal": {"enabled": True, "num_animals": 2}})
        config.validate()
        assert config.multi_animal.specimen_ids == ["specimen_0", "specimen_1"]

    def test_absent_block_leaves_the_feature_off(self):
        config = self.load({"mode": "singleview"})
        assert config.multi_animal.enabled is False

    def test_invalid_block_fails_validation(self):
        # load_config validates, so a bad multi-animal block aborts the run at
        # load time rather than at the first batch.
        with pytest.raises(MultiAnimalConfigError, match="camera_mode"):
            self.load(
                {
                    "mode": "multiview",
                    "multi_animal": {"enabled": True, "num_animals": 2, "camera_mode": "first_specimen"},
                }
            )

    def test_shipped_example_config_is_valid(self):
        path = os.path.join(EXAMPLES_DIR, "multiview_multianimal_mice.json")
        config = load_config(config_file=path)
        config.validate()
        assert config.multi_animal.enabled is True
        assert config.multi_animal.num_animals == 2
        assert config.multi_animal.specimen_ids == ["mouse_a", "mouse_b"]
        assert config.multi_animal.camera_mode == "scene_head"
