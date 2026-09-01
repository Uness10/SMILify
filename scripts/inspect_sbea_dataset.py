#!/usr/bin/env python3
"""Inspect an SBeA dataset (figshare 'SBeA upload data' / shank3b social videos)
and report how far it is from what SMILify's multi-view pipeline needs.

Usage
-----
  # 1. cheap: read the zip index only, nothing is extracted
  python inspect_sbea_dataset.py --zip SBeA_upload_data.zip

  # 2. deep: after extracting one session (or the shank3b sample)
  python inspect_sbea_dataset.py --dir /hpcwork/$USER/data/SBeA/sample

What SMILify wants per session (see README "Input directory structure"):
    session_xxx/
      calibration.toml            anipose format: size, matrix, distortions,
                                  rotation (rvec), translation  -- per camera
      points3d.h5                 anipose-format 3D keypoints
      Camera0/video.mp4 + video.mp4.predictions.slp
      Camera1/...

What SBeA ships per session (README_SBeA_tracker.md):
    [rec]-[A1A2]-[date]-caliParas.mat        MATLAB calibration
    [rec]-[A1A2]-[date]-camera-#.mp4|avi     one file per view
    [rec]-[A1A2]-[date]-raw3d.mat            3D skeletons, no identity
    [rec]-[A1A2]-[date]-rot3d.mat            rotated to world frame, no identity
    [rec]-[A1A2]-[date]-id3d.mat             rotated + identities   <-- the useful one
    [rec]-...-correctedresult.json           VIS masks
    [rec]-...-corrpredid.csv                 identity predictions
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict

# ----------------------------------------------------------------- helpers

ROLE_PATTERNS = [
    ("calibration", re.compile(r"caliParas.*\.mat$", re.I)),
    ("video", re.compile(r"camera[-_]?\d+.*\.(mp4|avi|mkv)$", re.I)),
    ("pose3d_id", re.compile(r"id3d.*\.mat$", re.I)),
    ("pose3d_rot", re.compile(r"rot3d.*\.mat$", re.I)),
    ("pose3d_raw", re.compile(r"raw3d.*\.mat$", re.I)),
    ("masks_json", re.compile(r"(corrected|raw)result.*\.json$", re.I)),
    ("identity_csv", re.compile(r"pred_?id.*\.csv$", re.I)),
    ("sleap", re.compile(r"\.(slp|analysis\.h5|predictions\.h5)$", re.I)),
    ("calibration_toml", re.compile(r"calibration\.toml$", re.I)),
]

SESSION_RE = re.compile(r"^(?P<session>.+?)-(?:camera-\d+|caliParas|raw3d|rot3d|id3d|"
                        r"rawresult|correctedresult|predid|corrpredid)", re.I)
CAM_RE = re.compile(r"camera[-_]?(\d+)", re.I)


def role_of(name: str) -> str | None:
    base = os.path.basename(name)
    for role, pat in ROLE_PATTERNS:
        if pat.search(base):
            return role
    return None


def session_of(name: str) -> str | None:
    m = SESSION_RE.match(os.path.basename(name))
    return m.group("session") if m else None


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def summarise(entries: list[tuple[str, int]]) -> None:
    """entries: (relative path, size in bytes)."""
    print(f"\n{len(entries)} files, {human(sum(s for _, s in entries))} total\n")

    ext = Counter(os.path.splitext(n)[1].lower() or "<none>" for n, _ in entries)
    print("by extension:")
    for e, c in ext.most_common(15):
        print(f"  {e:<16} {c}")

    tops = Counter(n.split("/")[0] for n, _ in entries if "/" in n)
    if tops:
        print("\ntop-level entries:")
        for t, c in tops.most_common(15):
            print(f"  {t:<40} {c} files")

    sessions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    orphans = []
    for name, _ in entries:
        role = role_of(name)
        if role is None:
            continue
        sess = session_of(name) or os.path.dirname(name) or "<root>"
        sessions[sess][role].append(name)
        if role is None:
            orphans.append(name)

    print(f"\ndetected sessions: {len(sessions)}")
    for sess in sorted(sessions)[:10]:
        roles = sessions[sess]
        cams = sorted({CAM_RE.search(os.path.basename(v)).group(1)
                       for v in roles.get("video", []) if CAM_RE.search(os.path.basename(v))})
        bits = [f"{len(v)}x {k}" for k, v in sorted(roles.items())]
        print(f"  {sess}")
        print(f"      cameras: {cams if cams else 'NONE'} | " + ", ".join(bits))
    if len(sessions) > 10:
        print(f"  ... and {len(sessions) - 10} more")

    checklist(sessions)


def checklist(sessions: dict) -> None:
    print("\n" + "=" * 72)
    print("SMILify multi-view readiness")
    print("=" * 72)

    if not sessions:
        print("  no SBeA-style sessions recognised -- check the naming pattern")
        return

    n = len(sessions)
    have = lambda role: sum(1 for s in sessions.values() if s.get(role))  # noqa: E731
    multiview = sum(
        1 for s in sessions.values()
        if len({CAM_RE.search(os.path.basename(v)).group(1)
                for v in s.get("video", []) if CAM_RE.search(os.path.basename(v))}) >= 2
    )

    rows = [
        (">=2 synchronised camera views", multiview, n,
         "-> Camera<N>/video.mp4"),
        ("camera calibration (caliParas.mat)", have("calibration"), n,
         "NEEDS CONVERSION -> calibration.toml (anipose: size/matrix/distortions/rotation/translation)"),
        ("3D poses with identity (id3d.mat)", have("pose3d_id"), n,
         "NEEDS CONVERSION -> points3d.h5"),
        ("3D poses, world frame (rot3d.mat)", have("pose3d_rot"), n, "fallback if id3d missing"),
        ("instance masks (correctedresult.json)", have("masks_json"), n,
         "optional: silhouette loss / per-animal crops"),
        ("per-view 2D keypoints (.slp/.h5)", have("sleap"), n,
         "ABSENT in SBeA -> run SLEAP yourself, or reproject id3d with generate_reprojections.py"),
        ("ready-made calibration.toml", have("calibration_toml"), n, "not expected from SBeA"),
    ]
    for label, got, tot, note in rows:
        mark = "ok  " if got == tot and got else ("part" if got else "MISS")
        print(f"  [{mark}] {label:<38} {got}/{tot}  {note}")

    print("""
Conversion route (all pieces already exist in the repo):
  1. caliParas.mat  -> calibration.toml   (write once; watch MATLAB conventions:
     matrices are transposed vs OpenCV, principal point is 1-based)
  2. id3d.mat       -> points3d.h5        (anipose layout; see
     smal_fitter/sleap_data/triangulate_3d_points.py:save_points3d_h5)
  3. 2D per view    -> smal_fitter/sleap_data/generate_reprojections.py
     (reproject the 3D onto each view) or run SLEAP on the videos
  4. split the two mice into one "specimen" stream each, using -id3d identities
     and the correctedresult.json masks for cropping
""")


# ----------------------------------------------------------------- zip mode

def inspect_zip(path: str) -> None:
    print(f"reading index of {path} (nothing is extracted)")
    with zipfile.ZipFile(path) as z:
        entries = [(i.filename, i.file_size) for i in z.infolist() if not i.is_dir()]
    summarise(entries)


# ----------------------------------------------------------------- dir mode

def inspect_dir(root: str) -> None:
    entries = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            try:
                entries.append((os.path.relpath(p, root), os.path.getsize(p)))
            except OSError:
                pass
    summarise(entries)
    deep(root, entries)


def _mat_load(path: str):
    """Load a .mat of either flavour. Returns (kind, object)."""
    try:
        import scipy.io as sio
        return "scipy", sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import h5py
        return "h5py", h5py.File(path, "r")
    except Exception as e:  # noqa: BLE001
        return "error", e


def _describe(obj, indent="      ", depth=0):
    import numpy as np
    if depth > 2:
        return
    if hasattr(obj, "_fieldnames"):                      # matlab struct
        for f in obj._fieldnames:
            v = getattr(obj, f)
            shape = getattr(v, "shape", None)
            print(f"{indent}{f}: {type(v).__name__}{'' if shape is None else ' ' + str(shape)}")
            if hasattr(v, "_fieldnames"):
                _describe(v, indent + "  ", depth + 1)
            elif isinstance(v, np.ndarray) and v.size <= 12:
                print(f"{indent}  = {np.array2string(v, precision=4)}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("__"):
                continue
            shape = getattr(v, "shape", None)
            print(f"{indent}{k}: {type(v).__name__}{'' if shape is None else ' ' + str(shape)}")
            _describe(v, indent + "  ", depth + 1)


def deep(root: str, entries) -> None:
    print("\n" + "=" * 72)
    print("deep inspection of extracted files")
    print("=" * 72)

    by_role = defaultdict(list)
    for name, _ in entries:
        r = role_of(name)
        if r:
            by_role[r].append(os.path.join(root, name))

    for role in ("calibration", "pose3d_id", "pose3d_rot", "pose3d_raw"):
        for path in by_role.get(role, [])[:2]:
            print(f"\n-- {role}: {os.path.relpath(path, root)}")
            kind, obj = _mat_load(path)
            if kind == "error":
                print(f"   could not read: {obj}")
                continue
            if kind == "h5py":
                obj.visit(lambda n: print(f"      {n}"))
                obj.close()
            else:
                _describe(obj)

    vids = by_role.get("video", [])
    if vids:
        print("\n-- videos")
        try:
            import cv2
            for path in vids[:8]:
                cap = cv2.VideoCapture(path)
                print(f"   {os.path.basename(path):<48} "
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                      f"@{cap.get(cv2.CAP_PROP_FPS):.1f}fps "
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
                cap.release()
            print("   -> frame counts must match across cameras of one session "
                  "(SMILify assumes synchronised views)")
        except ImportError:
            print("   opencv not available; skipping probe "
                  "(or run: ffprobe -v error -show_streams <file>)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip", help="SBeA_upload_data.zip (index is read, nothing extracted)")
    g.add_argument("--dir", help="extracted session folder or dataset root")
    a = ap.parse_args()
    if a.zip:
        inspect_zip(a.zip)
    else:
        inspect_dir(a.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
