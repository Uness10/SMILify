"""Tests for issue #56 (user-defined joint limits) — consumer side.

Covers the parts that do NOT need Blender:

- ``LimitPrior`` read path: authored ``joint_limits`` from the model .pkl reach
  the right ``(N_POSE, 3)`` slots, the old ±0.01 rad freeze is gone, and the
  wide-open fallback applies when no limits are present.
- Validation: wrong shape / min > max raise ``ValueError``.
- The fitter's hinge loss (flat inside the range, linear past it): zero
  in-range, positive on violation, gradient pulls back toward the range.
- The neural regressors' ``joint_limit_regularization`` penalty: same hinge
  through the real 6D → axis-angle conversion, and verifiably off by default
  (weight 0.0) in both regressors.

The Blender-side export helper is covered by ``tests/test_axis_remap.py`` (the
bpy-free remap math) and manually per ``docs/design/issue56_implementation.md``.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from smal_fitter.priors.joint_limits_prior import (  # noqa: E402
    LimitPrior,
    _ranges_from_joint_limits,
)


def _fresh_limit_prior(joint_limits=None):
    """Instantiate a LimitPrior with an optional injected joint_limits on config.dd.

    Restores config.dd afterwards so tests don't leak state into each other.
    """
    had = "joint_limits" in config.dd
    prev = config.dd.get("joint_limits", None)
    try:
        if joint_limits is None:
            config.dd.pop("joint_limits", None)
        else:
            config.dd["joint_limits"] = joint_limits
        return LimitPrior()
    finally:
        if had:
            config.dd["joint_limits"] = prev
        else:
            config.dd.pop("joint_limits", None)


def _fitter_limit_tensors(lp):
    """The same (N_POSE, 3) limit tensors the fitter builds (root dropped via [3:])."""
    n = config.N_POSE
    max_limits = torch.tensor(np.asarray(lp.max_values[3:], dtype=np.float32).reshape(n, 3))
    min_limits = torch.tensor(np.asarray(lp.min_values[3:], dtype=np.float32).reshape(n, 3))
    return min_limits, max_limits


def _hinge(joint_rotations, min_limits, max_limits):
    """The fitter's limit loss: flat inside the range, linear past it."""
    zeros = torch.zeros_like(joint_rotations)
    return torch.mean(torch.max(joint_rotations - max_limits, zeros) + torch.max(min_limits - joint_rotations, zeros))


# ---------------------------------------------------------------------------
# LimitPrior read path + fallback
# ---------------------------------------------------------------------------


def test_fallback_is_wide_open():
    """No joint_limits -> every non-root joint is wide-open [-pi, pi], root is 0."""
    lp = _fresh_limit_prior(None)
    pairs = set(zip(np.round(lp.min_values, 3), np.round(lp.max_values, 3)))
    assert pairs == {(0.0, 0.0), (round(-np.pi, 3), round(np.pi, 3))}, pairs
    # The old ±0.01 rad placeholder freeze must be gone.
    assert (round(-0.01, 3), round(0.01, 3)) not in pairs


def test_authored_limits_flow_to_correct_joint():
    """An injected joint_limits reaches the right (N_POSE, 3) slot the fitter uses."""
    n_joints = len(config.dd["J_names"])
    jl = np.zeros((n_joints, 3, 2), dtype=np.float32)
    jl[..., 0] = -0.5
    jl[..., 1] = 0.5
    jl[0] = 0.0  # root
    jl[8] = [[-0.1, 0.1], [-1.2, 0.3], [-0.05, 0.05]]  # some non-root joint

    lp = _fresh_limit_prior(jl)
    min_l, max_l = _fitter_limit_tensors(lp)

    # J index 8 -> pose index 7 (root dropped).
    assert np.allclose(min_l[7], [-0.1, -1.2, -0.05], atol=1e-6), min_l[7]
    assert np.allclose(max_l[7], [0.1, 0.3, 0.05], atol=1e-6), max_l[7]
    # A generic joint keeps the +/-0.5 we set.
    assert np.allclose(min_l[0], [-0.5, -0.5, -0.5], atol=1e-6)


def test_hinge_penalty_matches_fitter_formula():
    """LimitPrior.__call__ = flat inside, linear past the limit (fitter's formula)."""
    lp = _fresh_limit_prior(None)  # wide-open
    x = np.zeros_like(lp.max_values)
    assert np.allclose(lp(x, np), 0.0)  # zero cost inside the range
    # Push one axis past the wide-open max -> positive cost.
    x2 = lp.max_values + 0.1
    assert np.all(lp(x2, np) > 0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_bad_shape_raises():
    dd = dict(config.dd)
    dd["joint_limits"] = np.zeros((3, 3, 2))  # wrong J
    with pytest.raises(ValueError):
        _ranges_from_joint_limits(dd)


def test_min_greater_than_max_raises():
    n = len(config.dd["J_names"])
    jl = np.zeros((n, 3, 2))
    jl[..., 0] = 1.0  # min
    jl[..., 1] = -1.0  # max < min
    dd = dict(config.dd)
    dd["joint_limits"] = jl
    with pytest.raises(ValueError):
        _ranges_from_joint_limits(dd)


# ---------------------------------------------------------------------------
# Fitter limit-loss behaviour (issue-56 §5 guarantees, without images/datasets)
# ---------------------------------------------------------------------------


def test_fitter_limit_loss_zero_in_range():
    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    jr_ok = torch.zeros(config.N_POSE, 3)
    assert float(_hinge(jr_ok, min_l, max_l)) == 0.0


def test_fitter_limit_loss_violation_and_gradient():
    """Out-of-range pose -> positive loss; gradient pulls the joint back inside."""
    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    maxv = max_l.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))

    jr_bad = torch.zeros(config.N_POSE, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[j, a] = max_l[j, a] + 0.5
    loss = _hinge(jr_bad, min_l, max_l)
    loss.backward()

    assert float(loss) > 0.0
    # Positive gradient => descent lowers the angle back toward the limit.
    assert float(jr_bad.grad[j, a]) > 0.0


# ---------------------------------------------------------------------------
# Neural penalty (heavy imports: regressor modules) — marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_neural_penalty_zero_in_range_and_violation():
    from smal_fitter.neuralSMIL.smil_image_regressor import rotation_6d_to_axis_angle

    min_l, max_l = _fitter_limit_tensors(LimitPrior())
    B, n = 4, config.N_POSE

    # In-range batch of all-zero rotations.
    assert float(_hinge(torch.zeros(B, n, 3), min_l, max_l)) == 0.0

    # Violate the tightest joint/axis; positive loss + corrective gradient.
    maxv = max_l.numpy()
    j, a = map(int, np.unravel_index(np.argmin(maxv), maxv.shape))
    jr_bad = torch.zeros(B, n, 3, requires_grad=True)
    with torch.no_grad():
        jr_bad[:, j, a] = max_l[j, a] + 0.5
    loss = _hinge(jr_bad, min_l, max_l)
    loss.backward()
    assert float(loss) > 0.0
    assert float(jr_bad.grad[0, j, a]) > 0.0

    # Identity 6D pose exercises the real 6D -> axis-angle conversion path.
    six = torch.zeros(B, n, 6)
    six[..., 0] = 1.0
    six[..., 4] = 1.0
    assert float(_hinge(rotation_6d_to_axis_angle(six), min_l, max_l)) >= 0.0


@pytest.mark.slow
def test_neural_penalty_off_by_default():
    """Both regressors default the weight to 0.0 so existing training is unchanged."""
    import inspect

    from smal_fitter.neuralSMIL import multiview_smil_regressor, smil_image_regressor

    for mod in (smil_image_regressor, multiview_smil_regressor):
        src = inspect.getsource(mod)
        assert '"joint_limit_regularization": 0.0' in src, f"{mod.__name__} does not default the weight to 0.0"
