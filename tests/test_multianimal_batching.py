"""
Tests for the animal-axis tensor plumbing (``multianimal/batching.py``).

The flatten/expand helpers are what let the existing SMAL forward, renderer and
losses run unchanged on N specimens, so their *ordering* contract matters as
much as their shapes: ``flatten_animal_axis`` and ``expand_scene_to_animals``
must agree, or a specimen would be scored against another animal's camera.
"""

import pytest
import torch

from smal_fitter.neuralSMIL.multianimal.batching import (
    aggregate_specimen_losses,
    animal_mask_to_tensor,
    expand_scene_to_animals,
    flatten_animal_axis,
    masked_mean,
    merge_loss_components,
    select_specimen,
    stack_specimen_params,
    unflatten_animal_axis,
)


class TestFlattenRoundTrip:
    def test_round_trip_preserves_values(self):
        tensor = torch.randn(4, 3, 7, 2)
        flat = flatten_animal_axis(tensor)
        assert flat.shape == (12, 7, 2)
        assert torch.equal(unflatten_animal_axis(flat, 3), tensor)

    def test_row_major_ordering(self):
        # Sample-major: all of sample 0's specimens, then sample 1's.
        tensor = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]])  # (B=2, N=2, 1)
        assert flatten_animal_axis(tensor).flatten().tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_expand_matches_flatten_ordering(self):
        # A scene quantity expanded to N must line up row-for-row with the
        # flattened animal axis, otherwise specimen i would get sample j's camera.
        scene = torch.tensor([[10.0], [20.0]])  # (B=2, 1)
        animals = torch.tensor([[[0.0], [1.0]], [[2.0], [3.0]]])  # (B=2, N=2, 1)
        expanded = expand_scene_to_animals(scene, 2)
        flat = flatten_animal_axis(animals)
        assert expanded.shape[0] == flat.shape[0]
        assert expanded.flatten().tolist() == [10.0, 10.0, 20.0, 20.0]

    def test_expand_preserves_trailing_dims(self):
        scene = torch.randn(5, 3, 3)
        assert expand_scene_to_animals(scene, 4).shape == (20, 3, 3)

    def test_expand_n_equals_one_is_identity(self):
        scene = torch.randn(6, 9)
        assert torch.equal(expand_scene_to_animals(scene, 1), scene)

    def test_flatten_rejects_1d(self):
        with pytest.raises(ValueError, match="at least 2 dims"):
            flatten_animal_axis(torch.randn(5))

    def test_unflatten_rejects_indivisible(self):
        with pytest.raises(ValueError, match="not divisible"):
            unflatten_animal_axis(torch.randn(7, 2), 3)


class TestParamStacking:
    def test_stack_and_select_round_trip(self):
        per_specimen = [
            {"betas": torch.full((2, 5), float(i)), "trans": torch.full((2, 3), float(i))} for i in range(3)
        ]
        stacked = stack_specimen_params(per_specimen)
        assert stacked["betas"].shape == (2, 3, 5)
        for i in range(3):
            selected = select_specimen(stacked, i)
            assert torch.equal(selected["betas"], per_specimen[i]["betas"])

    def test_non_tensor_entries_are_skipped(self):
        per_specimen = [{"betas": torch.zeros(2, 5), "iteration_history": {"pose": []}} for _ in range(2)]
        stacked = stack_specimen_params(per_specimen)
        assert set(stacked) == {"betas"}

    def test_partially_present_keys_are_skipped(self):
        per_specimen = [{"betas": torch.zeros(2, 5)}, {}]
        assert stack_specimen_params(per_specimen) == {}


class TestAnimalMaskTensor:
    def test_pads_short_masks(self):
        mask = animal_mask_to_tensor([[True], [True, False]], num_animals=3)
        assert mask.tolist() == [[True, False, False], [True, False, False]]

    def test_truncates_long_masks(self):
        mask = animal_mask_to_tensor([[True, True, True]], num_animals=2)
        assert mask.tolist() == [[True, True]]


class TestMaskedMean:
    def test_matches_plain_mean_without_a_mask(self):
        values = torch.tensor([1.0, 2.0, 3.0])
        assert masked_mean(values).item() == pytest.approx(2.0)

    def test_ignores_masked_entries(self):
        values = torch.tensor([1.0, 100.0, 3.0])
        mask = torch.tensor([True, False, True])
        assert masked_mean(values, mask).item() == pytest.approx(2.0, abs=1e-5)

    def test_empty_mask_returns_zero_not_nan(self):
        values = torch.tensor([1.0, 2.0])
        result = masked_mean(values, torch.tensor([False, False]))
        assert result.item() == 0.0
        assert torch.isfinite(result)


class TestAggregation:
    def test_mean_over_present_specimens(self):
        losses = [torch.tensor(2.0), torch.tensor(4.0)]
        assert aggregate_specimen_losses(losses, [3, 3]).item() == pytest.approx(3.0)

    def test_absent_specimen_does_not_dilute(self):
        # An N=3 run on 2-mouse clips must train like a 2-mouse run.
        losses = [torch.tensor(2.0), torch.tensor(4.0), torch.tensor(0.0)]
        assert aggregate_specimen_losses(losses, [3, 3, 0]).item() == pytest.approx(3.0)

    def test_sum_reduction(self):
        losses = [torch.tensor(2.0), torch.tensor(4.0)]
        assert aggregate_specimen_losses(losses, [1, 1], reduction="sum").item() == pytest.approx(6.0)

    def test_weighted_mean_uses_presence_counts(self):
        losses = [torch.tensor(0.0), torch.tensor(10.0)]
        result = aggregate_specimen_losses(losses, [3, 1], reduction="weighted_mean")
        assert result.item() == pytest.approx(2.5)

    def test_all_absent_returns_a_differentiable_zero(self):
        losses = [(torch.tensor(1.0, requires_grad=True) * 2)]
        result = aggregate_specimen_losses(losses, [0])
        assert result.item() == 0.0
        assert result.requires_grad

    def test_gradient_flows_through_aggregation(self):
        param = torch.tensor(3.0, requires_grad=True)
        result = aggregate_specimen_losses([param * 2, param * 4], [1, 1])
        result.backward()
        assert param.grad.item() == pytest.approx(3.0)  # (2 + 4) / 2

    def test_unknown_reduction_is_rejected(self):
        with pytest.raises(ValueError, match="unknown reduction"):
            aggregate_specimen_losses([torch.tensor(1.0)], [1], reduction="median")

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            aggregate_specimen_losses([torch.tensor(1.0)], [1, 2])


class TestComponentMerging:
    def test_averages_shared_components_and_names_per_specimen_ones(self):
        components = [
            {"keypoint_2d": torch.tensor(1.0), "betas": torch.tensor(4.0)},
            {"keypoint_2d": torch.tensor(3.0), "betas": torch.tensor(6.0)},
        ]
        merged = merge_loss_components(components, [2, 2], specimen_ids=["mouse_a", "mouse_b"])

        assert merged["keypoint_2d"].item() == pytest.approx(2.0)
        assert merged["keypoint_2d/mouse_a"].item() == pytest.approx(1.0)
        assert merged["keypoint_2d/mouse_b"].item() == pytest.approx(3.0)
        assert merged["betas"].item() == pytest.approx(5.0)

    def test_component_present_for_only_one_specimen(self):
        components = [{"silhouette": torch.tensor(2.0)}, {}]
        merged = merge_loss_components(components, [1, 1], specimen_ids=["a", "b"])
        assert merged["silhouette"].item() == pytest.approx(2.0)
        assert "silhouette/b" not in merged

    def test_empty_input_returns_empty(self):
        assert merge_loss_components([], []) == {}
