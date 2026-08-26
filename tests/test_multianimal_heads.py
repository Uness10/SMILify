"""
Tests for the specimen head bank (``multianimal/heads.py``, ``specimen_heads.py``).

The head bank is where design doc §3-§5 lives: N copies of the existing head,
head i permanently bound to specimen i, new heads seeded from the pretrained
one.  These tests use a lightweight stub head so they run without pytorch3d, the
SMAL model pickle or a GPU -- the contract being tested is the bank's, not the
transformer's.
"""

import copy

import pytest
import torch
import torch.nn as nn

from smal_fitter.neuralSMIL.multianimal.heads import MLPParameterHead
from smal_fitter.neuralSMIL.multianimal.parameter_layout import ParameterLayout, parse_flat_parameter_vector
from smal_fitter.neuralSMIL.multianimal.specimen_heads import (
    REPLICATED,
    SHARED_QUERY,
    ReplicatedSpecimenHeads,
    SharedQuerySpecimenHeads,
    build_specimen_heads,
    specimen_parameter_groups,
)

FEATURE_DIM = 16
CONTEXT_DIM = 16


def make_layout(n_pose=4, betas=5, scales=0, joint_trans=0):
    return ParameterLayout(
        global_rot_dim=3,
        joint_rot_dim=n_pose * 3,
        betas_dim=betas,
        trans_dim=3,
        fov_dim=1,
        cam_rot_dim=9,
        cam_trans_dim=3,
        scales_dim=scales,
        joint_trans_dim=joint_trans,
        n_pose=n_pose,
        rotation_representation="axis_angle",
    )


class StubHead(nn.Module):
    """Minimal head with the required ``(features, spatial) -> dict`` signature."""

    def __init__(self, feature_dim=FEATURE_DIM, out_dim=7):
        super().__init__()
        self.linear = nn.Linear(feature_dim, out_dim)
        self.context_proj = nn.Linear(CONTEXT_DIM, out_dim)

    def forward(self, features, spatial_features=None):
        out = self.linear(features)
        if spatial_features is not None:
            out = out + self.context_proj(spatial_features.mean(dim=1))
        return {"betas": out, "trans": out[:, :3]}


def stub_factory():
    torch.manual_seed(0)
    return StubHead()


class TestParameterLayout:
    def test_total_dim_is_the_sum_of_the_groups(self):
        layout = make_layout()
        assert layout.total_dim == 3 + 12 + 5 + 3 + 1 + 9 + 3

    def test_parse_splits_groups_in_layout_order(self):
        layout = make_layout()
        output = torch.arange(layout.total_dim, dtype=torch.float32).unsqueeze(0)
        params = parse_flat_parameter_vector(output, layout)

        assert params["global_rot"].tolist() == [[0.0, 1.0, 2.0]]
        assert params["joint_rot"].shape == (1, 4, 3)
        assert params["betas"].shape == (1, 5)
        assert params["cam_rot"].shape == (1, 3, 3)

    def test_camera_can_be_excluded_without_shifting_later_groups(self):
        layout = make_layout(scales=5, joint_trans=5)
        output = torch.arange(layout.total_dim, dtype=torch.float32).unsqueeze(0)

        with_camera = parse_flat_parameter_vector(output, layout, include_camera=True)
        without_camera = parse_flat_parameter_vector(output, layout, include_camera=False)

        assert "cam_rot" not in without_camera
        # The groups *after* the camera must land on the same values either way.
        assert torch.equal(with_camera["log_beta_scales"], without_camera["log_beta_scales"])
        assert torch.equal(with_camera["betas_trans"], without_camera["betas_trans"])

    def test_width_mismatch_is_reported(self):
        layout = make_layout()
        with pytest.raises(ValueError, match="does not match layout"):
            parse_flat_parameter_vector(torch.zeros(1, layout.total_dim + 1), layout)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            parse_flat_parameter_vector(torch.zeros(2, 3, 4), make_layout())

    def test_per_joint_scales_are_reshaped(self):
        layout = ParameterLayout(
            global_rot_dim=3,
            joint_rot_dim=12,
            betas_dim=5,
            trans_dim=3,
            fov_dim=1,
            cam_rot_dim=9,
            cam_trans_dim=3,
            scales_dim=9,
            joint_trans_dim=0,
            n_pose=4,
            scales_use_pca=False,
        )
        params = parse_flat_parameter_vector(torch.zeros(2, layout.total_dim), layout)
        assert params["log_beta_scales"].shape == (2, 3, 3)


class TestMLPParameterHead:
    def test_output_shapes(self):
        layout = make_layout()
        head = MLPParameterHead(FEATURE_DIM, hidden_dim=32, layout=layout)
        params = head(torch.randn(4, FEATURE_DIM))
        assert params["global_rot"].shape == (4, 3)
        assert params["joint_rot"].shape == (4, 4, 3)
        assert params["cam_rot"].shape == (4, 3, 3)

    def test_ignores_spatial_features(self):
        layout = make_layout()
        head = MLPParameterHead(FEATURE_DIM, hidden_dim=32, layout=layout).eval()
        features = torch.randn(2, FEATURE_DIM)
        with torch.no_grad():
            a = head(features)["betas"]
            b = head(features, torch.randn(2, 5, CONTEXT_DIM))["betas"]
        assert torch.equal(a, b)

    def test_loads_weights_saved_from_the_inline_head(self):
        layout = make_layout()
        head = MLPParameterHead(FEATURE_DIM, hidden_dim=32, layout=layout)
        # A single-animal checkpoint stores the head as top-level regressor keys.
        inline_state = {key: value.clone() for key, value in head.state_dict().items()}

        fresh = MLPParameterHead(FEATURE_DIM, hidden_dim=32, layout=layout)
        fresh.load_inline_head_state(inline_state)

        features = torch.randn(3, FEATURE_DIM)
        head.eval(), fresh.eval()
        with torch.no_grad():
            assert torch.allclose(head(features)["betas"], fresh(features)["betas"])

    def test_missing_key_is_reported(self):
        layout = make_layout()
        head = MLPParameterHead(FEATURE_DIM, hidden_dim=32, layout=layout)
        state = dict(head.state_dict())
        del state["fc2.weight"]
        with pytest.raises(KeyError, match="fc2.weight"):
            head.load_inline_head_state(state)


class TestReplicatedSpecimenHeads:
    def test_returns_one_dict_per_specimen(self):
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=3)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        assert len(outputs) == 3
        assert all(out["betas"].shape == (2, 7) for out in outputs)

    def test_heads_are_independent_modules(self):
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=3)
        assert len({id(head) for head in heads.heads}) == 3

    def test_tied_init_starts_every_head_identical(self):
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=3, tie_first_head_init=True)
        features = torch.randn(2, FEATURE_DIM)
        outputs = heads(features)
        assert torch.allclose(outputs[0]["betas"], outputs[1]["betas"])
        assert torch.allclose(outputs[0]["betas"], outputs[2]["betas"])

    def test_heads_diverge_once_trained_separately(self):
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=2)
        with torch.no_grad():
            heads.heads[1].linear.weight.add_(1.0)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        assert not torch.allclose(outputs[0]["betas"], outputs[1]["betas"])

    def test_gradients_reach_every_head(self):
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=3)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        sum(out["betas"].sum() for out in outputs).backward()
        for head in heads.heads:
            assert head.linear.weight.grad is not None
            assert head.linear.weight.grad.abs().sum() > 0

    def test_specimen_zero_gradient_does_not_leak_into_specimen_one(self):
        # Strict head/specimen binding: supervising specimen 0 must never move
        # specimen 1's weights.
        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=2)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        outputs[0]["betas"].sum().backward()
        assert heads.heads[0].linear.weight.grad.abs().sum() > 0
        assert heads.heads[1].linear.weight.grad is None

    def test_seeding_from_a_pretrained_head(self):
        pretrained = StubHead()
        with torch.no_grad():
            pretrained.linear.weight.fill_(0.5)

        heads = ReplicatedSpecimenHeads(stub_factory, num_animals=3)
        heads.load_single_animal_head_state(pretrained.state_dict())

        for head in heads.heads:
            assert torch.allclose(head.linear.weight, torch.full_like(head.linear.weight, 0.5))

    def test_strategy_name(self):
        assert ReplicatedSpecimenHeads(stub_factory, 2).strategy == REPLICATED

    def test_rejects_zero_animals(self):
        with pytest.raises(ValueError, match="num_animals must be >= 1"):
            ReplicatedSpecimenHeads(stub_factory, 0)


class TestSharedQuerySpecimenHeads:
    def test_returns_one_dict_per_specimen(self):
        heads = SharedQuerySpecimenHeads(stub_factory, 3, feature_dim=FEATURE_DIM, context_dim=CONTEXT_DIM)
        outputs = heads(torch.randn(2, FEATURE_DIM), torch.randn(2, 5, CONTEXT_DIM))
        assert len(outputs) == 3
        assert all(out["betas"].shape == (2, 7) for out in outputs)

    def test_specimens_differ_thanks_to_the_embedding(self):
        heads = SharedQuerySpecimenHeads(stub_factory, 3, feature_dim=FEATURE_DIM, context_dim=CONTEXT_DIM)
        with torch.no_grad():  # make the embedding clearly non-degenerate
            heads.specimen_embedding.weight.copy_(torch.eye(3, FEATURE_DIM) * 5.0)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        assert not torch.allclose(outputs[0]["betas"], outputs[1]["betas"])

    def test_single_head_is_shared_across_specimens(self):
        heads = SharedQuerySpecimenHeads(stub_factory, 3, feature_dim=FEATURE_DIM)
        assert sum(1 for _ in heads.head.parameters()) == sum(1 for _ in stub_factory().parameters())

    def test_batch_ordering_matches_the_flatten_contract(self):
        # Sample b's specimen n must come from row b*N+n of the folded batch.
        heads = SharedQuerySpecimenHeads(stub_factory, 2, feature_dim=FEATURE_DIM)
        with torch.no_grad():
            heads.specimen_embedding.weight.zero_()
            heads.context_embedding.weight.zero_()
        features = torch.randn(3, FEATURE_DIM)
        outputs = heads(features)
        direct = heads.head(features)["betas"]
        assert torch.allclose(outputs[0]["betas"], direct, atol=1e-6)

    def test_gradient_reaches_the_embeddings(self):
        heads = SharedQuerySpecimenHeads(stub_factory, 3, feature_dim=FEATURE_DIM)
        outputs = heads(torch.randn(2, FEATURE_DIM))
        sum(out["betas"].sum() for out in outputs).backward()
        assert heads.specimen_embedding.weight.grad.abs().sum() > 0

    def test_parameter_count_is_far_below_the_replicated_bank(self):
        shared = SharedQuerySpecimenHeads(stub_factory, 8, feature_dim=FEATURE_DIM, context_dim=CONTEXT_DIM)
        replicated = ReplicatedSpecimenHeads(stub_factory, 8)
        assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in replicated.parameters())

    def test_strategy_name(self):
        heads = SharedQuerySpecimenHeads(stub_factory, 2, feature_dim=FEATURE_DIM)
        assert heads.strategy == SHARED_QUERY


class TestFactory:
    @pytest.mark.parametrize("strategy", [REPLICATED, SHARED_QUERY])
    def test_builds_both_strategies(self, strategy):
        heads = build_specimen_heads(strategy, stub_factory, 2, FEATURE_DIM, CONTEXT_DIM)
        assert heads.num_animals == 2
        assert len(heads(torch.randn(2, FEATURE_DIM))) == 2

    def test_unknown_strategy_names_the_valid_options(self):
        with pytest.raises(ValueError, match="replicated"):
            build_specimen_heads("detr_queries", stub_factory, 2, FEATURE_DIM)

    def test_describe_is_informative(self):
        heads = build_specimen_heads(REPLICATED, stub_factory, 3, FEATURE_DIM)
        text = heads.describe()
        assert "replicated" in text and "N=3" in text


class TestParameterGroups:
    def test_single_group_by_default(self):
        heads = ReplicatedSpecimenHeads(stub_factory, 3)
        groups = specimen_parameter_groups(heads, base_lr=1e-4)
        assert len(groups) == 1
        assert groups[0]["lr"] == pytest.approx(1e-4)

    def test_per_specimen_scaling(self):
        heads = ReplicatedSpecimenHeads(stub_factory, 3)
        groups = specimen_parameter_groups(heads, 1e-4, per_specimen_lr_scale=[1.0, 2.0, 0.5])
        assert [group["lr"] for group in groups] == pytest.approx([1e-4, 2e-4, 5e-5])

    def test_scale_length_mismatch_is_rejected(self):
        heads = ReplicatedSpecimenHeads(stub_factory, 3)
        with pytest.raises(ValueError, match="expected 3"):
            specimen_parameter_groups(heads, 1e-4, per_specimen_lr_scale=[1.0, 2.0])

    def test_deepcopy_of_the_bank_is_independent(self):
        heads = ReplicatedSpecimenHeads(stub_factory, 2)
        clone = copy.deepcopy(heads)
        with torch.no_grad():
            clone.heads[0].linear.weight.fill_(9.0)
        assert not torch.allclose(heads.heads[0].linear.weight, clone.heads[0].linear.weight)
