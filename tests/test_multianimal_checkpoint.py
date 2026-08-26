"""
Tests for checkpoint migration (``multianimal/checkpoint.py``).

The promise being tested is the one the design doc makes in §3 and the one that
makes multi-animal training affordable: an existing single-animal checkpoint can
seed every specimen head, the backbone loads untouched, and going back to the
single-animal model is possible.
"""

import pytest
import torch
import torch.nn as nn

from smal_fitter.neuralSMIL.multianimal.checkpoint import (
    MULTI_REPLICATED_PREFIX,
    MULTI_SHARED_PREFIX,
    SINGLE_TRANSFORMER_PREFIX,
    count_specimen_heads,
    detect_layout,
    filter_optimizer_state,
    specimen_head_keys,
    summarize_specimen_heads,
    to_multi_animal,
    to_single_animal,
)


def single_animal_transformer_state():
    """A single-animal checkpoint: backbone + one transformer head."""
    return {
        "backbone.model.blocks.0.weight": torch.ones(4, 4),
        "backbone.model.norm.bias": torch.zeros(4),
        f"{SINGLE_TRANSFORMER_PREFIX}pose_head.weight": torch.full((6, 8), 2.0),
        f"{SINGLE_TRANSFORMER_PREFIX}pose_head.bias": torch.full((6,), 3.0),
        f"{SINGLE_TRANSFORMER_PREFIX}betas_head.weight": torch.full((5, 8), 4.0),
    }


def single_animal_mlp_state():
    return {
        "backbone.model.conv1.weight": torch.ones(2, 2),
        "fc1.weight": torch.full((8, 4), 1.0),
        "fc1.bias": torch.zeros(8),
        "ln1.weight": torch.ones(8),
        "regressor.weight": torch.full((10, 2), 5.0),
    }


class TestLayoutDetection:
    def test_single_transformer(self):
        assert detect_layout(single_animal_transformer_state()) == "single_transformer"

    def test_single_mlp(self):
        assert detect_layout(single_animal_mlp_state()) == "single_mlp"

    def test_multi_replicated(self):
        state, _ = to_multi_animal(single_animal_transformer_state(), 3)
        assert detect_layout(state) == "multi_replicated"

    def test_multi_shared(self):
        state, _ = to_multi_animal(single_animal_transformer_state(), 3, head_strategy="shared_query")
        assert detect_layout(state) == "multi_shared"

    def test_unknown_layout(self):
        assert detect_layout({"backbone.weight": torch.zeros(1)}) == "unknown"

    def test_summary(self):
        state, _ = to_multi_animal(single_animal_transformer_state(), 3)
        assert summarize_specimen_heads(state) == {"layout": "multi_replicated", "num_specimen_heads": 3}


class TestSingleToMulti:
    def test_every_head_is_seeded_from_the_pretrained_one(self):
        source = single_animal_transformer_state()
        migrated, report = to_multi_animal(source, 3)

        for index in range(3):
            key = f"{MULTI_REPLICATED_PREFIX}{index}.pose_head.weight"
            assert torch.equal(migrated[key], source[f"{SINGLE_TRANSFORMER_PREFIX}pose_head.weight"])
        assert report.heads_seeded == [0, 1, 2]

    def test_seeded_heads_are_independent_tensors(self):
        # A shared storage would make every head move together on the first step.
        migrated, _ = to_multi_animal(single_animal_transformer_state(), 2)
        a = migrated[f"{MULTI_REPLICATED_PREFIX}0.pose_head.weight"]
        b = migrated[f"{MULTI_REPLICATED_PREFIX}1.pose_head.weight"]
        a.add_(1.0)
        assert not torch.equal(a, b)

    def test_backbone_keys_are_carried_through_untouched(self):
        source = single_animal_transformer_state()
        migrated, _ = to_multi_animal(source, 3)
        assert torch.equal(migrated["backbone.model.blocks.0.weight"], source["backbone.model.blocks.0.weight"])
        assert "backbone.model.norm.bias" in migrated

    def test_old_head_keys_are_gone(self):
        migrated, _ = to_multi_animal(single_animal_transformer_state(), 2)
        assert not any(key.startswith(SINGLE_TRANSFORMER_PREFIX) for key in migrated)

    def test_mlp_head_migrates(self):
        migrated, _ = to_multi_animal(single_animal_mlp_state(), 2)
        assert f"{MULTI_REPLICATED_PREFIX}0.fc1.weight" in migrated
        assert f"{MULTI_REPLICATED_PREFIX}1.regressor.weight" in migrated
        assert "fc1.weight" not in migrated

    def test_shared_query_target_writes_a_single_head(self):
        migrated, report = to_multi_animal(single_animal_transformer_state(), 3, head_strategy="shared_query")
        assert f"{MULTI_SHARED_PREFIX}pose_head.weight" in migrated
        assert not any(key.startswith(MULTI_REPLICATED_PREFIX) for key in migrated)
        assert report.heads_seeded == [0]

    def test_n_equals_one_is_a_faithful_rename(self):
        source = single_animal_transformer_state()
        migrated, _ = to_multi_animal(source, 1)
        assert torch.equal(
            migrated[f"{MULTI_REPLICATED_PREFIX}0.betas_head.weight"],
            source[f"{SINGLE_TRANSFORMER_PREFIX}betas_head.weight"],
        )

    def test_unknown_checkpoint_is_rejected_with_guidance(self):
        with pytest.raises(ValueError, match="no recognisable parameter head"):
            to_multi_animal({"something.else": torch.zeros(1)}, 2)

    def test_invalid_num_animals_is_rejected(self):
        with pytest.raises(ValueError, match="num_animals must be >= 1"):
            to_multi_animal(single_animal_transformer_state(), 0)

    def test_unknown_strategy_is_rejected(self):
        with pytest.raises(ValueError, match="unknown head_strategy"):
            to_multi_animal(single_animal_transformer_state(), 2, head_strategy="detr")


class TestMultiToMulti:
    def test_growing_n_keeps_existing_heads_and_seeds_the_rest(self):
        two_heads, _ = to_multi_animal(single_animal_transformer_state(), 2)
        two_heads[f"{MULTI_REPLICATED_PREFIX}1.pose_head.weight"].fill_(99.0)

        three_heads, report = to_multi_animal(two_heads, 3)

        assert report.heads_kept == [0, 1]
        assert report.heads_seeded == [2]
        # head 1 kept its trained (distinct) weights
        assert three_heads[f"{MULTI_REPLICATED_PREFIX}1.pose_head.weight"].flatten()[0].item() == pytest.approx(99.0)
        # head 2 was seeded from the donor (head 0)
        assert torch.equal(
            three_heads[f"{MULTI_REPLICATED_PREFIX}2.pose_head.weight"],
            three_heads[f"{MULTI_REPLICATED_PREFIX}0.pose_head.weight"],
        )

    def test_shrinking_n_reports_the_dropped_heads(self):
        three_heads, _ = to_multi_animal(single_animal_transformer_state(), 3)
        _, report = to_multi_animal(three_heads, 2)
        assert report.heads_dropped == [2]
        assert any("dropped" in warning for warning in report.warnings)

    def test_donor_head_can_be_chosen(self):
        two_heads, _ = to_multi_animal(single_animal_transformer_state(), 2)
        two_heads[f"{MULTI_REPLICATED_PREFIX}1.pose_head.weight"].fill_(7.0)

        grown, _ = to_multi_animal(two_heads, 3, donor_head=1)
        assert grown[f"{MULTI_REPLICATED_PREFIX}2.pose_head.weight"].flatten()[0].item() == pytest.approx(7.0)

    def test_missing_donor_falls_back_and_warns(self):
        two_heads, _ = to_multi_animal(single_animal_transformer_state(), 2)
        _, report = to_multi_animal(two_heads, 4, donor_head=9)
        assert any("donor head 9" in warning for warning in report.warnings)

    def test_count_specimen_heads(self):
        state, _ = to_multi_animal(single_animal_transformer_state(), 4)
        assert count_specimen_heads(state) == 4


class TestMultiToSingle:
    def test_specimen_zero_is_written_back_to_single_animal_keys(self):
        source = single_animal_transformer_state()
        multi, _ = to_multi_animal(source, 3)
        back, report = to_single_animal(multi)

        assert torch.equal(
            back[f"{SINGLE_TRANSFORMER_PREFIX}pose_head.weight"],
            source[f"{SINGLE_TRANSFORMER_PREFIX}pose_head.weight"],
        )
        assert report.heads_kept == [0]

    def test_a_chosen_specimen_can_be_exported(self):
        multi, _ = to_multi_animal(single_animal_transformer_state(), 3)
        multi[f"{MULTI_REPLICATED_PREFIX}2.pose_head.weight"].fill_(42.0)
        back, _ = to_single_animal(multi, specimen_index=2)
        assert back[f"{SINGLE_TRANSFORMER_PREFIX}pose_head.weight"].flatten()[0].item() == pytest.approx(42.0)

    def test_round_trip_is_lossless_for_specimen_zero(self):
        source = single_animal_transformer_state()
        multi, _ = to_multi_animal(source, 3)
        back, _ = to_single_animal(multi)
        for key, value in source.items():
            assert key in back
            assert torch.equal(back[key], value)

    def test_single_animal_input_is_rejected(self):
        with pytest.raises(ValueError, match="expected a multi-animal checkpoint"):
            to_single_animal(single_animal_transformer_state())


class TestLoadIntoModel:
    class TinyMultiAnimalModel(nn.Module):
        """Stands in for a real regressor: a backbone plus a 2-head bank."""

        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.model = nn.Module()
            self.backbone.model.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False)])
            self.specimen_heads = nn.Module()
            self.specimen_heads.heads = nn.ModuleList([nn.Linear(8, 6) for _ in range(2)])
            self.specimen_heads.strategy = "replicated"
            self.num_animals = 2

    def test_single_animal_checkpoint_seeds_both_heads(self):
        from smal_fitter.neuralSMIL.multianimal.checkpoint import load_into_model

        model = self.TinyMultiAnimalModel()
        state = {
            f"{SINGLE_TRANSFORMER_PREFIX}weight": torch.full((6, 8), 0.25),
            f"{SINGLE_TRANSFORMER_PREFIX}bias": torch.full((6,), 0.5),
        }
        report = load_into_model(model, state)

        assert report.heads_seeded == [0, 1]
        for head in model.specimen_heads.heads:
            assert torch.allclose(head.weight, torch.full_like(head.weight, 0.25))
        assert "single_transformer -> multi_replicated" in report.summary()


class TestHelpers:
    def test_optimizer_state_is_dropped_by_default(self):
        assert filter_optimizer_state({"state": {}}) is None

    def test_optimizer_state_can_be_kept_explicitly(self):
        state = {"state": {}}
        assert filter_optimizer_state(state, keep=True) is state

    def test_specimen_head_key_expansion(self):
        keys = specimen_head_keys(2, ["pose_head.weight"])
        assert keys == [
            f"{MULTI_REPLICATED_PREFIX}0.pose_head.weight",
            f"{MULTI_REPLICATED_PREFIX}1.pose_head.weight",
        ]
