"""End-to-end check: synthesise known anatomical angles, run the analysis, recover them."""
import subprocess, sys, os, csv, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
PKL = sys.argv[1]
subprocess.run([sys.executable, f"{HERE}/make_synthetic_export.py", "--smal-file", PKL,
                "--out", "/tmp/rt"], check=True, capture_output=True)
subprocess.run([sys.executable, f"{HERE}/analyse_gait.py", "--npz", "/tmp/rt.npz",
                "--smal-file", PKL, "--label", "RT", "--out", "/tmp/rtfigs"],
               check=True, capture_output=True)
rows = list(csv.DictReader(open("/tmp/rtfigs/angles_RT.csv")))
AMP = {"R1": (22.0, 32.0), "R2": (20.0, -6.0), "R3": (11.0, -36.0),
       "L1": (22.0, 32.0), "L2": (20.0, -6.0), "L3": (11.0, -36.0)}
fails = 0
for leg, (amp, mid) in AMP.items():
    a = np.array([float(r[f"{leg}_ThC_alpha_protraction_abs"]) for r in rows])
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    got_mid, got_amp = (hi + lo) / 2, (hi - lo) / 2
    ok = abs(got_mid - mid) < 1.5 and abs(got_amp - amp) < 1.5
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {leg} ThC alpha: midpoint {got_mid:+6.1f} (want {mid:+6.1f}),"
          f" half-range {got_amp:5.1f} (want {amp:5.1f})")
# the fused trochanter-femur joint must carry no rotation
fe = np.array([[float(r[f"{leg}_fe_hinge_flexion_rel"]) for leg in AMP] for r in rows])
print(f"{'PASS' if np.abs(fe).max() < 1.0 else 'FAIL'} fused trochanter-femur stays at "
      f"{np.abs(fe).max():.2f} deg")
# hinge residual must be ~0 for a true hinge
res = np.array([[float(r[f"{leg}_ti_offaxis_residual"]) for leg in AMP] for r in rows])
print(f"{'PASS' if np.abs(res).max() < 1.0 else 'FAIL'} FTi off-hinge-axis residual max "
      f"{np.abs(res).max():.2f} deg")
sys.exit(1 if fails else 0)
