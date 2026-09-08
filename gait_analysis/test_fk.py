"""Verify smil_fk against the repo's own torch implementation and the rest pose."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smil_fk

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
PKL = os.path.join(REPO, "3D_model_prep", "SMILy_STICK.pkl")

m = smil_fk.load_model(PKL)
print(f"model: {m.n_joints} joints, {m.shapedirs.shape[2]} betas, static={m.static_joint_locs}")

rng = np.random.default_rng(0)
F = 7
poses = rng.normal(0, 0.35, (F, m.n_joints, 3))
betas = rng.normal(0, 0.5, (m.shapedirs.shape[2],))
lbs = rng.normal(0, 0.05, (F, m.n_joints, 3))
btr = rng.normal(0, 0.01, (F, m.n_joints, 3))

# ---- 1. zero pose reproduces the rest joints exactly
J0, _, Jrest = smil_fk.forward_kinematics(m, np.zeros((1, m.n_joints, 3)), betas=betas)
assert np.abs(J0[0] - Jrest[0]).max() < 1e-9, np.abs(J0[0] - Jrest[0]).max()
print("PASS zero pose == rest joints")

# ---- 2. rest joints from betas match J_regressor @ v_shaped
v = m.v_template + m.shapedirs @ betas
assert np.abs(m.J_regressor @ v - Jrest[0]).max() < 1e-9
print("PASS shape-dependent rest joints")

# ---- 3. match the repo's torch batch_global_rigid_transformation
try:
    import torch, types
    fake = types.ModuleType("config"); fake.ALLOW_LIMB_SCALING = True; fake.DEBUG = False
    sys.modules.setdefault("config", fake)
    sys.path.insert(0, REPO)
    from smal_model.batch_lbs import batch_global_rigid_transformation, batch_rodrigues

    for prop in (False, True):
        Rt = batch_rodrigues(torch.tensor(poses.reshape(-1, 3), dtype=torch.float32)).reshape(F, m.n_joints, 3, 3)
        Jt = torch.tensor(np.repeat(m.rest_joints(betas)[None], F, 0), dtype=torch.float32)
        newJ, _ = batch_global_rigid_transformation(
            Rt, Jt, torch.tensor(m.parents),
            betas_logscale=torch.tensor(lbs, dtype=torch.float32), betas_trans=torch.tensor(btr, dtype=torch.float32),
            propagate_scaling=prop, num_joints=m.n_joints,
        )
        mine, _, _ = smil_fk.forward_kinematics(
            m, poses, betas=betas, log_beta_scales=lbs, betas_trans=btr,
            propagate_scaling=prop, zero_global=False,
        )
        err = np.abs(newJ.numpy() - mine).max()
        assert err < 2e-5, err
        print(f"PASS matches torch batch_global_rigid_transformation (propagate_scaling={prop}, max err {err:.2e})")

    Rn = smil_fk.rodrigues(poses)
    assert np.abs(Rn - Rt.numpy().reshape(F, m.n_joints, 3, 3)).max() < 1e-5
    print("PASS rodrigues matches batch_rodrigues")
except ImportError as e:
    print("SKIP torch cross-check:", e)

print("\nall FK checks passed")
