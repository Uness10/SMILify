#!/usr/bin/env python3
"""Write a synthetic animation export in the exact schema of --export_animation.

Used to smoke-test analyse_gait.py end to end without a checkpoint, and to show
what the figures look like before the real run. Step period, duty factor and the
tripod phasing are the values measured off the SMILySTICKS footage
(find_gait_windows.py): period 42 frames, tripod groups {R1,L2,R3} / {R2,L3,L1}.
"""
import argparse, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smil_fk, anatomical as ana

ap = argparse.ArgumentParser()
ap.add_argument("--smal-file", required=True)
ap.add_argument("--out", default="synthetic_export")
ap.add_argument("--frames", type=int, default=500)
ap.add_argument("--period", type=float, default=42.0)
ap.add_argument("--fps", type=float, default=60.0)
a = ap.parse_args()

m = smil_fk.load_model(a.smal_file)
F, T = a.frames, a.period
rng = np.random.default_rng(7)

PHASE = {(1, "r"): 0.0, (2, "l"): 0.05, (3, "r"): 0.10,       # tripod group A
         (2, "r"): 0.52, (3, "l"): 0.60, (1, "l"): 0.50}      # tripod group B
# protraction amplitude / midpoint per leg pair, from Guschlbauer et al. 2022 walk ranges
AMP = {1: (22.0, 32.0), 2: (20.0, -6.0), 3: (11.0, -36.0)}     # (half-range, midpoint) deg
LEV = {1: 7.0, 2: 8.0, 3: 6.0}
FTI = {1: (14.0, -30.0), 2: (16.0, -34.0), 3: (12.0, -26.0)}   # (half-range, offset from rest) deg
CTR = 9.0

poses = np.zeros((F, m.n_joints, 3))
t = np.arange(F)
for leg in ana.LEGS:
    n, s = leg
    ph = 2 * np.pi * (t / T - PHASE[leg])
    # asymmetric wave: slow retraction, fast protraction (stance:swing ~ 0.7:0.3)
    duty = 0.68
    u = ((t / T - PHASE[leg]) % 1.0)
    saw = np.where(u < duty, 1 - u / duty, (u - duty) / (1 - duty))   # 1 -> 0 -> 1
    amp, mid = AMP[n]
    alpha = mid + amp * (2 * saw - 1)
    beta = LEV[n] * np.clip(np.sin(np.pi * np.clip((u - duty) / (1 - duty), 0, 1)), 0, None)
    fa, fo = FTI[n]
    gamma_rel = fo + fa * np.sin(2 * np.pi * (u - 0.15))
    ctr_rel = CTR * np.sin(2 * np.pi * (u - 0.05))

    # build the ThC rotation the same way anatomical.thc_intrinsic reads it back:
    # R = Rz(a) Rx(b) Ry(c) in the model frame, with the per-side sign flips.
    sgn = 1.0 if s == "r" else -1.0
    co, tr, ti = (m.joint(ana.jname(n, s, seg)) for seg in ("co", "tr", "ti"))
    rest = ana.rest_angles(m, n, s)
    eul = np.stack([sgn * (alpha - rest["alpha"]), -sgn * (beta - rest["beta"]),
                    np.zeros(F)], 1)
    poses[:, co, :] = Rotation.from_euler("ZXY", eul, degrees=True).as_rotvec()
    poses[:, tr, :] = np.radians(ctr_rel)[:, None] * ana.hinge_axis(m, n, s, "tr")[None, :]
    poses[:, ti, :] = np.radians(gamma_rel)[:, None] * ana.hinge_axis(m, n, s, "ti")[None, :]

poses[:, 0, 2] = np.radians(2.0 * np.sin(2 * np.pi * t / (3 * T)))     # slight yaw
poses += rng.normal(0, 0.004, poses.shape)                             # regressor jitter

trans = np.stack([t / T * 0.35, np.zeros(F), np.zeros(F)], 1).astype(np.float32)
betas = rng.normal(0, 0.2, m.shapedirs.shape[2]).astype(np.float32)

np.savez(a.out + ".npz",
         poses=poses.astype(np.float32), trans=trans,
         betas=betas, betas_per_frame=np.repeat(betas[None], F, 0),
         log_beta_scales=np.zeros((F, m.n_joints, 3), np.float32),
         fps=np.float32(a.fps))
json.dump({"schema_version": "1.1", "n_frames": F, "n_joints": m.n_joints,
           "joint_names": m.joint_names, "parents": m.parents.tolist(),
           "rotation_representation": "axis_angle", "root_joint_index": 0,
           "fps": a.fps, "cameras": []}, open(a.out + ".json", "w"), indent=2)
print(f"wrote {a.out}.npz / .json  ({F} frames, {m.n_joints} joints)")
