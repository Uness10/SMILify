#!/usr/bin/env python3
"""
Probe an extracted SBeA release (SBM-VIS and/or SBeA_upload_data) and report the
facts the SMILify converter depends on.

This answers, empirically:
  1. What is the directory layout, and does it match the tracker README's
     "[F1]-[F2]-[F3]-camera-N.avi + -caliParas.mat" convention?
  2. Are raw multi-view videos present, or only processed results?
  3. What variables live inside caliParas.mat  -> becomes calibration.toml
  4. What variables live inside id3d.mat       -> becomes points3d.h5 "tracks"
  5. What do the 2D pose CSVs and YouTubeVIS JSONs look like?

Run on the login node inside the SMILify env:
    conda activate "$HPCWORK/conda_envs/pytorch3d"
    python diagnostics/sbea_dataset_PROBE.py /hpcwork/<user>/datasets

Output is deliberately bounded so it can be pasted back in full.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

MAX_EXAMPLES = 6          # example filenames printed per extension
MAX_TREE_DIRS = 60        # directories printed in the layout section
TREE_DEPTH = 3
INTERESTING = {".mat", ".avi", ".mp4", ".csv", ".json", ".h5", ".toml", ".npy", ".pkl"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n:.1f}TB"


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------- inventory

def walk(root: Path):
    """Single pass: extension counts, sizes, examples, and the directory set."""
    ext_count: Counter = Counter()
    ext_bytes: Counter = Counter()
    ext_examples: dict[str, list[Path]] = defaultdict(list)
    dirs: list[Path] = []
    n_files = 0

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        if depth <= TREE_DEPTH:
            dirs.append(d)
        for fn in sorted(filenames):
            p = d / fn
            ext = p.suffix.lower()
            ext_count[ext] += 1
            n_files += 1
            try:
                ext_bytes[ext] += p.stat().st_size
            except OSError:
                pass
            if len(ext_examples[ext]) < MAX_EXAMPLES:
                ext_examples[ext].append(p)

    return ext_count, ext_bytes, ext_examples, dirs, n_files


def report_inventory(root: Path) -> dict[str, list[Path]]:
    ext_count, ext_bytes, ext_examples, dirs, n_files = walk(root)

    rule(f"1. INVENTORY  —  {root}")
    print(f"{n_files} files, {human(sum(ext_bytes.values()))} total\n")
    print(f"{'ext':<10}{'count':>9}{'size':>12}")
    print("-" * 31)
    for ext, cnt in ext_count.most_common(20):
        print(f"{ext or '(none)':<10}{cnt:>9}{human(ext_bytes[ext]):>12}")

    rule(f"2. LAYOUT  (directories, depth <= {TREE_DEPTH})")
    shown = dirs[:MAX_TREE_DIRS]
    for d in shown:
        rel = d.relative_to(root)
        depth = len(rel.parts)
        name = rel.parts[-1] if rel.parts else "."
        print(f"{'  ' * depth}{name}/")
    if len(dirs) > MAX_TREE_DIRS:
        print(f"  ... and {len(dirs) - MAX_TREE_DIRS} more directories")

    rule("3. EXAMPLE FILENAMES  (does the naming match the tracker README?)")
    for ext in sorted(INTERESTING):
        if ext not in ext_examples:
            continue
        print(f"\n[{ext}]  {ext_count[ext]} file(s)")
        for p in ext_examples[ext]:
            try:
                sz = human(p.stat().st_size)
            except OSError:
                sz = "?"
            print(f"   {p.relative_to(root)}   ({sz})")

    return ext_examples


# ------------------------------------------------------------------- .mat

def describe_mat(path: Path) -> None:
    """Dump variable names, shapes and dtypes from a .mat of either vintage."""
    print(f"\n--- {path.name}")
    print(f"    {path}")

    # MATLAB v7.3 is HDF5; anything older is not.
    try:
        import h5py

        with h5py.File(path, "r") as f:
            print("    format: MAT v7.3 (HDF5)")

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"      {name:<40} shape={obj.shape} dtype={obj.dtype}")

            f.visititems(visit)
            return
    except (OSError, ImportError):
        pass

    try:
        from scipy.io import loadmat
    except ImportError:
        print("    !! scipy not available — cannot read pre-v7.3 MAT")
        return

    try:
        md = loadmat(path, squeeze_me=False, struct_as_record=False)
    except Exception as exc:  # noqa: BLE001 — report, never abort the probe
        print(f"    !! could not read: {exc}")
        return

    print("    format: MAT <= v7.2 (scipy)")
    for k, v in md.items():
        if k.startswith("__"):
            continue
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v).__name__)
        print(f"      {k:<40} shape={shape} dtype={dtype}")
        # One level into struct arrays — this is where camera params usually hide.
        if getattr(v, "dtype", None) is not None and v.dtype == object and v.size:
            try:
                inner = v.flat[0]
                fields = getattr(inner, "_fieldnames", None)
                if fields:
                    print(f"        struct fields: {fields}")
                    for fname in fields[:12]:
                        fv = getattr(inner, fname)
                        print(f"          .{fname:<28} shape={getattr(fv, 'shape', None)} "
                              f"dtype={getattr(fv, 'dtype', type(fv).__name__)}")
            except Exception:  # noqa: BLE001
                pass


def report_mats(root: Path, examples: dict[str, list[Path]]) -> None:
    rule("4. MAT CONTENTS  (calibration + 3D poses)")

    mats = examples.get(".mat", [])
    if not mats:
        print("No .mat files found. If the bundle only ships processed results,")
        print("this is the finding that changes the plan — say so and we re-scope.")
        return

    # Prefer the two files the converter actually needs.
    wanted = ("calipara", "id3d", "rot3d", "raw3d")
    picked: list[Path] = []
    for key in wanted:
        for dirpath, _dirnames, filenames in os.walk(root):
            hit = next((f for f in sorted(filenames)
                        if f.lower().endswith(".mat") and key in f.lower()), None)
            if hit:
                picked.append(Path(dirpath) / hit)
                break

    if not picked:
        print("None of caliParas/id3d/rot3d/raw3d found by name — showing the first .mat instead.")
        picked = mats[:2]

    for p in picked:
        describe_mat(p)


# ------------------------------------------------------------- csv / json

def report_tabular(root: Path, examples: dict[str, list[Path]]) -> None:
    rule("5. CSV HEADERS  (2D poses / identity predictions)")
    csvs = examples.get(".csv", [])
    if not csvs:
        print("No CSV files found.")
    for p in csvs[:3]:
        print(f"\n--- {p.relative_to(root)}")
        try:
            with open(p, "r", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 4:
                        break
                    print(f"    {line.rstrip()[:220]}")
        except OSError as exc:
            print(f"    !! {exc}")

    rule("6. JSON STRUCTURE  (YouTubeVIS contours / keypoints)")
    jsons = examples.get(".json", [])
    if not jsons:
        print("No JSON files found.")
    for p in jsons[:3]:
        print(f"\n--- {p.relative_to(root)}  ({human(p.stat().st_size)})")
        try:
            with open(p, "r", errors="replace") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"    !! could not parse: {exc}")
            continue

        if isinstance(data, dict):
            print(f"    top-level keys: {list(data.keys())[:15]}")
            for key in ("categories", "info", "licenses"):
                if key in data:
                    print(f"    {key}: {json.dumps(data[key])[:400]}")
            for key in ("videos", "annotations", "images"):
                if key in data and isinstance(data[key], list):
                    print(f"    {key}: {len(data[key])} entries")
                    if data[key]:
                        first = data[key][0]
                        if isinstance(first, dict):
                            print(f"      [0] keys: {list(first.keys())}")
        elif isinstance(data, list):
            print(f"    list of {len(data)}")
            if data and isinstance(data[0], dict):
                print(f"      [0] keys: {list(data[0].keys())}")


# ------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".", help="extracted dataset directory")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    examples = report_inventory(root)
    report_mats(root, examples)
    report_tabular(root, examples)

    rule("DONE")
    print("Paste this whole report back. The .mat variable names and shapes in")
    print("section 4 are what determine the converter; everything else is layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
