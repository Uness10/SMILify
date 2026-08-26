"""
Tests for per-specimen loss aggregation (``multianimal/losses.py``).

Two properties carry real correctness risk and are checked here directly:

* the camera (a scene-level quantity) must be supervised **once**, not N times,
  or its weight is silently multiplied by N relative to every body term;
* an absent or heavily occluded specimen must contribute no gradient and must
  not dilute the average.
"""

import pytest
import torch

from smal_fitter.neuralSMIL.multianimal.losses import (
    MultiAnimalLossAggregator,
    apply_visibility_floor,
    presence_counts_from_mask,
    weights_for_specimen,
)

BASE_WEIGHTS = {
    "global_rot": 0.02,
    "joint_rot": 0.02,
    "betas": 0.01,
    "keypoint_2d": 1.0,
    "fov": 0.5,
    "cam_rot": 0.5,
    "cam_trans": 0.5,
}


class TestWeightsForSpecimen:
    def test_specimen_zero_keeps_the_camera_weights(self):
        weights = weights_for_specimen(BASE_WEIGHTS, 0)
        assert weights["fov"] == pytest.approx(0.5)
        assert weights["cam_rot"] == pytest.approx(0.5)

    def test_later_specimens_have_the_camera_zeroed(self):
        for index in (1, 2):
            weights = weights_for_specimen(BASE_WEIGHTS, index)
            assert weights["fov"] == 0.0
            assert weights["cam_rot"] == 0.0
            assert weights["cam_trans"] == 0.0

    def test_body_weights_are_untouched(self):
        weights = weights_for_specimen(BASE_WEIGHTS, 2)
        assert weights["joint_rot"] == pytest.approx(0.02)
        assert weights["keypoint_2d"] == pytest.approx(1.0)

    def test_input_is_not_mutated(self):
        original = dict(BASE_WEIGHTS)
        weights_for_specimen(BASE_WEIGHTS, 1)
        assert BASE_WEIGHTS == original

    def test_missing_camera_keys_are_not_invented(self):
        weights = weights_for_specimen({"betas": 1.0}, 1)
        assert weights == {"betas": 1.0}


class TestPresenceCounts:
    def test_counts_per_specimen(self):
        mask = torch.tensor([[True, True, False], [True, False, False]])
        assert presence_counts_from_mask(mask) == [2, 1, 0]

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(B, N\)"):
            presence_counts_from_mask(torch.tensor([True, False]))


class TestVisibilityFloor:
    def test_disabled_floor_is_a_no_op(self):
        mask = torch.tensor([[True, True]])
        assert torch.equal(apply_visibility_floor(mask, torch.zeros_like(mask, dtype=torch.long), 0), mask)

    def test_heavily_occluded_specimen_becomes_absent(self):
        mask = torch.tensor([[True, True, True]])
        counts = torch.tensor([[12, 2, 7]])
        assert apply_visibility_floor(mask, counts, 5).tolist() == [[True, False, True]]

    def test_never_resurrects_an_absent_specimen(self):
        mask = torch.tensor([[False, True]])
        counts = torch.tensor([[99, 99]])
        assert apply_visibility_floor(mask, counts, 5).tolist() == [[False, True]]

    def test_shape_mismatch_is_reported(self):
        with pytest.raises(ValueError, match="visibility shape"):
            apply_visibility_floor(torch.tensor([[True]]), torch.tensor([[1, 2]]), 1)


class TestAggregator:
    def make(self, num_animals=3, **kwargs):
        return MultiAnimalLossAggregator(
            num_animals=num_animals,
            specimen_ids=[f"mouse_{i}" for i in range(num_animals)],
            **kwargs,
        )

    def test_averages_over_present_specimens(self):
        aggregator = self.make()
        losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]

        def loss_fn(index, weights):
            return losses[index], {"betas": losses[index]}

        total = aggregator(loss_fn, BASE_WEIGHTS, [4, 4, 4])
        assert total.item() == pytest.approx(2.0)

    def test_absent_specimen_is_never_evaluated(self):
        aggregator = self.make()
        seen = []

        def loss_fn(index, weights):
            seen.append(index)
            return torch.tensor(1.0), {}

        aggregator(loss_fn, BASE_WEIGHTS, [4, 0, 4])
        assert seen == [0, 2]

    def test_absent_specimen_can_be_evaluated_when_requested(self):
        aggregator = self.make(drop_absent=False)
        seen = []

        def loss_fn(index, weights):
            seen.append(index)
            return torch.tensor(1.0), {}

        aggregator(loss_fn, BASE_WEIGHTS, [4, 0, 4])
        assert seen == [0, 1, 2]

    def test_camera_weight_reaches_only_specimen_zero(self):
        aggregator = self.make()
        seen = {}

        def loss_fn(index, weights):
            seen[index] = weights["fov"]
            return torch.tensor(1.0), {}

        aggregator(loss_fn, BASE_WEIGHTS, [1, 1, 1])
        assert seen == {0: 0.5, 1: 0.0, 2: 0.0}

    def test_components_carry_both_averaged_and_per_specimen_entries(self):
        aggregator = self.make(num_animals=2)

        def loss_fn(index, weights):
            value = torch.tensor(float(index) + 1.0)
            return value, {"keypoint_2d": value}

        total, components = aggregator(loss_fn, BASE_WEIGHTS, [3, 3], return_components=True)

        assert total.item() == pytest.approx(1.5)
        assert components["keypoint_2d"].item() == pytest.approx(1.5)
        assert components["keypoint_2d/mouse_0"].item() == pytest.approx(1.0)
        assert components["keypoint_2d/mouse_1"].item() == pytest.approx(2.0)
        assert components["num_specimens_supervised"].item() == pytest.approx(2.0)

    def test_all_absent_returns_a_finite_zero(self):
        aggregator = self.make()

        def loss_fn(index, weights):  # pragma: no cover - must not be called
            raise AssertionError("absent specimens must not be evaluated")

        total, components = aggregator(loss_fn, BASE_WEIGHTS, [0, 0, 0], return_components=True)
        assert total.item() == 0.0
        assert torch.isfinite(total)
        assert components == {}

    def test_gradient_flows_to_every_present_specimen(self):
        aggregator = self.make(num_animals=2)
        params = [torch.tensor(1.0, requires_grad=True), torch.tensor(1.0, requires_grad=True)]

        def loss_fn(index, weights):
            value = params[index] * 3.0
            return value, {}

        aggregator(loss_fn, BASE_WEIGHTS, [1, 1]).backward()
        assert params[0].grad.item() == pytest.approx(1.5)
        assert params[1].grad.item() == pytest.approx(1.5)

    def test_specimen_returning_none_is_skipped(self):
        aggregator = self.make(num_animals=2)

        def loss_fn(index, weights):
            return None if index == 1 else (torch.tensor(4.0), {})

        assert aggregator(loss_fn, BASE_WEIGHTS, [1, 1]).item() == pytest.approx(4.0)

    def test_presence_length_mismatch_is_rejected(self):
        aggregator = self.make(num_animals=3)
        with pytest.raises(ValueError, match="expected num_animals=3"):
            aggregator(lambda i, w: (torch.tensor(0.0), {}), BASE_WEIGHTS, [1, 1])

    def test_specimen_ids_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="expected num_animals=3"):
            MultiAnimalLossAggregator(num_animals=3, specimen_ids=["a", "b"])
