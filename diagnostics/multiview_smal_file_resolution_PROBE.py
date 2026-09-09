#!/usr/bin/env python3
"""Static + behavioural probe for checkpoint-driven SMAL model resolution in
run_multiview_inference.py.

Regression guard for: a checkpoint trained with a non-default SMAL/SMIL model
(e.g. SMILy_STICK.pkl, N_BETAS=5) failed to load against config.py's default
(N_BETAS=13) with

    size mismatch for transformer_head.betas_head.weight:
        copying a param with shape [5, 1024] ... current model is [13, 1024]

because the multi-view script only applied --smal_file from the CLI and ignored
the smal_file that train_multiview_regressor.py records in the checkpoint
specifically for inference. The single-view script already read it.

Checks, without importing torch/pytorch3d:
  * the checkpoint is read once, before the SMAL override, which is itself
    before dataset and model construction (config.dd / N_POSE / N_BETAS must be
    correct at model-construction time, and the head widths derive from them);
  * the pre-loaded checkpoint is threaded into the model loader so a multi-GB
    file is not read twice;
  * precedence: --smal_file > checkpoint's smal_file > config.py default, same
    for shape_family;
  * a recorded path that does not resolve on this machine warns and continues,
    while an explicitly passed missing --smal_file raises;
  * a surviving size mismatch names the checkpoint's smal_file, the currently
    loaded one, and the fix.

Usage:  python diagnostics/multiview_smal_file_resolution_PROBE.py
"""

import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "smal_fitter", "neuralSMIL"))

import ast
import io
import sys

src = io.open("run_multiview_inference.py", encoding="utf-8").read()
tree = ast.parse(src)
ok = True


def check(c, m):
    global ok
    print(("PASS  " if c else "FAIL  ") + m)
    ok = ok and c


top = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
check("load_checkpoint_and_config" in top, "defined: load_checkpoint_and_config")
check("resolve_smal_file_for_checkpoint" in top, "defined: resolve_smal_file_for_checkpoint")

mi = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main_inference")
body_src = ast.get_source_segment(src, mi)

# ordering: checkpoint read + SMAL override must precede dataset and model construction
i_ckpt = body_src.index("load_checkpoint_and_config(")
i_smal = body_src.index("resolve_smal_file_for_checkpoint(")
i_ds = body_src.index("SLEAPMultiViewDataset(")
i_model = body_src.index("load_multiview_model_from_checkpoint(")
check(i_ckpt < i_smal < i_ds < i_model, "order: read checkpoint -> apply SMAL override -> build dataset -> build model")

check(body_src.count("_find_default_checkpoint()") == 1, "checkpoint path resolved exactly once")
check("checkpoint=checkpoint" in body_src, "pre-loaded checkpoint is reused (file not read twice)")
check(src.count("torch.load(str(checkpoint_path)") == 2, "torch.load call sites: helper + guarded fallback only")

# the loader must tolerate a pre-loaded checkpoint
ld = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_multiview_model_from_checkpoint")
check("checkpoint" in [a.arg for a in ld.args.args], "loader accepts a pre-loaded checkpoint")

check("size mismatch" in src, "size-mismatch failures explain the smal_file cause")
check("N_BETAS={config.N_BETAS}" in src, "the error reports the currently loaded N_BETAS")

# behavioural: exercise resolve_smal_file_for_checkpoint with stubs
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_smal_file_for_checkpoint")
import types
import typing

calls = []
cfgmod = types.SimpleNamespace(SHAPE_FAMILY=0, N_POSE=1, N_BETAS=13, SMAL_FILE="default.pkl")
ns = {
    "Optional": typing.Optional,
    "config": cfgmod,
    "Path": __import__("pathlib").Path,
    "apply_smal_file_override": lambda f, shape_family=None: calls.append((f, shape_family)),
}
exec(compile(ast.Module(body=[node], type_ignores=[]), "<x>", "exec"), ns)
fn = ns["resolve_smal_file_for_checkpoint"]

real = "/tmp/_fake_smal.pkl"
open(real, "w").close()

calls.clear()
fn({"smal_file": real, "shape_family": -1}, None, None)
check(calls == [(real, -1)], f"checkpoint's smal_file is applied when no CLI override: {calls}")

calls.clear()
fn({"smal_file": real, "shape_family": -1}, real, 7)
check(calls == [(real, 7)], f"--smal_file/--shape_family override the checkpoint: {calls}")

calls.clear()
fn({}, None, None)
check(calls == [], "no smal_file anywhere -> config.py default left alone")

calls.clear()
fn({"smal_file": "/nope/missing.pkl"}, None, None)
check(calls == [], "unresolvable checkpoint path warns instead of applying")

try:
    fn({}, "/nope/missing.pkl", None)
    raised = False
except FileNotFoundError:
    raised = True
check(raised, "explicitly passed --smal_file that is missing raises")

calls.clear()
fn({"smal_file": real}, None, None)
check(calls == [(real, 0)], f"missing shape_family falls back to config.SHAPE_FAMILY: {calls}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
