#!/usr/bin/env python3
"""Freeze the train/val/test split shared by both arms of the joint-limit prior study.

Phase 0 of ``scripts/prior_study/ROADMAP.md``. The study compares a multi-view (MV)
and a single-view (SV) arm, and the comparison is only apples-to-apples if both arms
evaluate on the *same underlying frames*. This script computes that split once, writes
it to ``prior_study_results/eval_split.json``, and can later re-verify that a run's
split determinants still reproduce it.

Why this is safe to do offline (probed, not assumed)
----------------------------------------------------
Both trainers split at the **sample** level with the same mechanism, so the split is a
pure function of ``(n_samples, seed, train_ratio, val_ratio)``:

* multi-view -- ``train_multiview_regressor.py:2390-2397``::

      total_size = len(dataset)                      # == metadata.attrs["num_samples"]
      train_size = int(total_size * config["train_ratio"])
      val_size   = int(total_size * config["val_ratio"])
      test_size  = total_size - train_size - val_size
      random_split(dataset, [train_size, val_size, test_size],
                   generator=torch.Generator().manual_seed(config["seed"]))

* single-view-from-multiview -- ``train_smil_regressor.py:1666-1687``::

      n_samples = int(dataset.num_samples)           # same attr, same number
      n_train = int(n_samples * train_ratio); n_val = int(n_samples * val_ratio)
      n_test  = n_samples - n_train - n_val
      random_split(range(n_samples), [n_train, n_val, n_test],
                   generator=torch.Generator().manual_seed(seed))
      # then expanded to view-items via dataset.item_sample_indices

``torch.utils.data.random_split`` draws ``randperm(sum(lengths), generator=...)``, which
depends only on the seed and the total -- not on the dataset object. So the two calls
above yield **identical sample partitions** whenever the seed and ratios match. Today
they do not: the MV stick config uses ``seed: 42`` and the SV example uses ``seed: 1234``.
Unifying the seed is the whole point of this step; ``tests/test_prior_study_eval_split.py``
pins the equivalence so it cannot silently regress.

The SV arm additionally needs the *view-item* indices for its test samples. Those come
from ``multiview_images/view_mask`` in the HDF5, expanded exactly as
``SLEAPMultiViewDataset._build_single_view_index`` does (row-major over
``(sample, view_slot)`` for every ``view_mask=True`` slot).

Usage
-----
::

    # freeze (run once, commit the JSON)
    python scripts/prior_study/freeze_eval_split.py \
        --dataset SMILySTICKS_centred_reprojected_FIXED.h5 --seed 42

    # re-verify later (exit 1 on any drift)
    python scripts/prior_study/freeze_eval_split.py \
        --dataset SMILySTICKS_centred_reprojected_FIXED.h5 --verify

    # validate the logic with no data on disk
    python scripts/prior_study/freeze_eval_split.py --self-test

Programmatic use from either arm::

    from scripts.prior_study.freeze_eval_split import load_eval_split
    split = load_eval_split()                       # default path
    split.assert_compatible(n_samples=len(dataset), seed=cfg["seed"],
                            train_ratio=..., val_ratio=...)
    test_samples = split.test_samples               # MV: dataset indices
    test_items   = split.test_view_items            # SV: view-item indices
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Repo root = .../SMILify (this file lives at scripts/prior_study/)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "prior_study_results" / "eval_split.json"

# Canonical study constants (pre-registered in prior_study_results/PROTOCOL.md).
# Changing any of these invalidates every frozen result; the verify path enforces it.
CANONICAL_SEED = 42
CANONICAL_TRAIN_RATIO = 0.85
CANONICAL_VAL_RATIO = 0.05
CANONICAL_TEST_RATIO = 0.10

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# split computation
# --------------------------------------------------------------------------- #
def split_sizes(n_samples: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    """Sample counts per split, matching both trainers' arithmetic exactly.

    Both use ``int(n * ratio)`` (truncation, not rounding) for train and val and give
    the remainder to test. Reimplementing this with ``round`` would silently shift the
    boundary by a sample or two.
    """
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    if n_test < 0:
        raise ValueError(f"train_ratio + val_ratio > 1 for n_samples={n_samples}")
    return n_train, n_val, n_test


def compute_sample_split(
    n_samples: int,
    seed: int = CANONICAL_SEED,
    train_ratio: float = CANONICAL_TRAIN_RATIO,
    val_ratio: float = CANONICAL_VAL_RATIO,
) -> Dict[str, List[int]]:
    """Reproduce the trainers' sample-level split.

    Returns sorted index lists. Order within a split is irrelevant downstream (the
    training loader shuffles and the test loader is a set), and sorting makes the JSON
    diffable and set comparison trivial.
    """
    import torch  # local import: the module is importable for its constants without torch

    n_train, n_val, n_test = split_sizes(n_samples, train_ratio, val_ratio)
    train, val, test = torch.utils.data.random_split(
        range(n_samples),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(int(seed)),
    )
    return {
        "train": sorted(int(i) for i in train),
        "val": sorted(int(i) for i in val),
        "test": sorted(int(i) for i in test),
    }


def build_view_item_index(view_mask: np.ndarray) -> np.ndarray:
    """``item_sample_indices``: the parent sample index of each single-view item.

    Mirrors ``SLEAPMultiViewDataset._build_single_view_index`` -- row-major over
    ``(sample, view_slot)`` for every ``view_mask=True`` slot. Item *k* of the
    single-view dataset belongs to sample ``result[k]``.
    """
    view_mask = np.asarray(view_mask, dtype=bool)
    if view_mask.ndim != 2:
        raise ValueError(f"view_mask must be (N, max_views), got shape {view_mask.shape}")
    sample_idx, _view_idx = np.nonzero(view_mask)  # np.nonzero is row-major
    return sample_idx.astype(np.int64)


def expand_samples_to_view_items(sample_indices: Sequence[int], item_sample_indices: np.ndarray) -> List[int]:
    """Map a set of sample indices to the single-view item indices they own."""
    wanted = np.zeros(int(item_sample_indices.max()) + 1 if len(item_sample_indices) else 0, dtype=bool)
    for s in sample_indices:
        if 0 <= int(s) < len(wanted):
            wanted[int(s)] = True
    if len(wanted) == 0:
        return []
    return [int(i) for i in np.nonzero(wanted[item_sample_indices])[0]]


# --------------------------------------------------------------------------- #
# dataset probing
# --------------------------------------------------------------------------- #
@dataclass
class DatasetFingerprint:
    """Cheap, split-relevant identity of the HDF5.

    Deliberately not a hash of the whole file (multi-GB, minutes to read). ``view_mask``
    is (N, max_views) bool -- kilobytes -- and it is exactly the array that determines
    both ``n_samples`` and the single-view item enumeration, so hashing it catches every
    change that could move the split.
    """

    path: str
    num_samples: int
    max_views: int
    n_joints: int
    file_size_bytes: int
    view_mask_sha256: str
    num_view_items: int


def probe_dataset(h5_path: Path) -> Tuple[DatasetFingerprint, np.ndarray]:
    """Read split determinants from the multi-view HDF5. Returns (fingerprint, item_sample_indices)."""
    import h5py

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        md = f["metadata"].attrs
        num_samples = int(md["num_samples"])
        max_views = int(md["max_views"])
        n_joints = int(md["n_joints"])
        view_mask = f["multiview_images/view_mask"][:]

    if view_mask.shape[0] != num_samples:
        raise ValueError(
            f"view_mask has {view_mask.shape[0]} rows but metadata says num_samples={num_samples}; "
            "the HDF5 is inconsistent and the split cannot be trusted."
        )

    item_sample_indices = build_view_item_index(view_mask)
    fp = DatasetFingerprint(
        path=str(h5_path),
        num_samples=num_samples,
        max_views=max_views,
        n_joints=n_joints,
        file_size_bytes=h5_path.stat().st_size,
        view_mask_sha256=hashlib.sha256(np.ascontiguousarray(view_mask, dtype=bool).tobytes()).hexdigest(),
        num_view_items=int(len(item_sample_indices)),
    )
    return fp, item_sample_indices


# --------------------------------------------------------------------------- #
# the frozen split object
# --------------------------------------------------------------------------- #
@dataclass
class EvalSplit:
    """A frozen split, as persisted to ``eval_split.json``."""

    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    num_samples: int
    train_samples: List[int]
    val_samples: List[int]
    test_samples: List[int]
    # Single-view arm: item indices into the expanded (sample, view) dataset.
    train_view_items: List[int] = field(default_factory=list)
    val_view_items: List[int] = field(default_factory=list)
    test_view_items: List[int] = field(default_factory=list)
    dataset: Optional[Dict] = None
    created_utc: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    # -- integrity ---------------------------------------------------------- #
    def assert_compatible(
        self,
        n_samples: int,
        seed: int,
        train_ratio: float = CANONICAL_TRAIN_RATIO,
        val_ratio: float = CANONICAL_VAL_RATIO,
    ) -> None:
        """Raise unless a run's split determinants reproduce this frozen split.

        Call this at the top of any study run. It is the guard that turns "we think both
        arms saw the same test frames" into something the code enforces.
        """
        problems = []
        if int(n_samples) != int(self.num_samples):
            problems.append(f"num_samples {n_samples} != frozen {self.num_samples}")
        if int(seed) != int(self.seed):
            problems.append(f"seed {seed} != frozen {self.seed}")
        if abs(float(train_ratio) - float(self.train_ratio)) > 1e-9:
            problems.append(f"train_ratio {train_ratio} != frozen {self.train_ratio}")
        if abs(float(val_ratio) - float(self.val_ratio)) > 1e-9:
            problems.append(f"val_ratio {val_ratio} != frozen {self.val_ratio}")
        if problems:
            raise ValueError(
                "Run does not reproduce the frozen prior-study split:\n  - " + "\n  - ".join(problems) + "\n"
                "Either fix the config or re-freeze deliberately (and note it in PROTOCOL.md)."
            )
        recomputed = compute_sample_split(n_samples, seed, train_ratio, val_ratio)
        if recomputed["test"] != list(self.test_samples):
            raise ValueError(
                "Split determinants match but the recomputed test set differs from the frozen one. "
                "This means torch's random_split RNG changed across versions -- do NOT proceed; "
                "results from before and after are not comparable."
            )

    # -- (de)serialisation -------------------------------------------------- #
    def to_dict(self) -> Dict:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "num_samples": self.num_samples,
            "counts": {
                "train_samples": len(self.train_samples),
                "val_samples": len(self.val_samples),
                "test_samples": len(self.test_samples),
                "train_view_items": len(self.train_view_items),
                "val_view_items": len(self.val_view_items),
                "test_view_items": len(self.test_view_items),
            },
            "dataset": self.dataset,
            "samples": {
                "train": self.train_samples,
                "val": self.val_samples,
                "test": self.test_samples,
            },
            "view_items": {
                "train": self.train_view_items,
                "val": self.val_view_items,
                "test": self.test_view_items,
            },
            "notes": {
                "sample_indices": (
                    "Indices into the multi-view dataset (== HDF5 sample order). The MV arm's "
                    "test Subset contains exactly these; sorted here for diffability."
                ),
                "view_items": (
                    "Indices into the expanded single-view dataset built by "
                    "SLEAPMultiViewDataset with return_single_view=True, expand_all_views=True. "
                    "Row-major over (sample, view_slot) for every view_mask=True slot."
                ),
                "provenance": (
                    "Reproduces torch.utils.data.random_split(range(n), [n_train, n_val, n_test], "
                    "generator=torch.Generator().manual_seed(seed)) -- the exact call made by "
                    "train_multiview_regressor.py and train_smil_regressor.py."
                ),
            },
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EvalSplit":
        return cls(
            seed=int(d["seed"]),
            train_ratio=float(d["train_ratio"]),
            val_ratio=float(d["val_ratio"]),
            test_ratio=float(d["test_ratio"]),
            num_samples=int(d["num_samples"]),
            train_samples=[int(i) for i in d["samples"]["train"]],
            val_samples=[int(i) for i in d["samples"]["val"]],
            test_samples=[int(i) for i in d["samples"]["test"]],
            train_view_items=[int(i) for i in d.get("view_items", {}).get("train", [])],
            val_view_items=[int(i) for i in d.get("view_items", {}).get("val", [])],
            test_view_items=[int(i) for i in d.get("view_items", {}).get("test", [])],
            dataset=d.get("dataset"),
            created_utc=d.get("created_utc"),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
        )

    def write(self, path: Path = DEFAULT_OUT) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def load_eval_split(path: Path = DEFAULT_OUT) -> EvalSplit:
    """Load the frozen split. Both arms should call this rather than re-deriving indices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No frozen split at {path}. Run:\n  python scripts/prior_study/freeze_eval_split.py --dataset <sticks.h5>"
        )
    return EvalSplit.from_dict(json.loads(path.read_text(encoding="utf-8")))


def build_eval_split(
    n_samples: int,
    seed: int = CANONICAL_SEED,
    train_ratio: float = CANONICAL_TRAIN_RATIO,
    val_ratio: float = CANONICAL_VAL_RATIO,
    item_sample_indices: Optional[np.ndarray] = None,
    dataset_info: Optional[Dict] = None,
) -> EvalSplit:
    """Compute a fresh :class:`EvalSplit` (does not write it)."""
    s = compute_sample_split(n_samples, seed, train_ratio, val_ratio)
    test_ratio = round(1.0 - train_ratio - val_ratio, 10)

    if item_sample_indices is not None:
        isi = np.asarray(item_sample_indices, dtype=np.int64)
        tr = expand_samples_to_view_items(s["train"], isi)
        va = expand_samples_to_view_items(s["val"], isi)
        te = expand_samples_to_view_items(s["test"], isi)
        # Every item must land in exactly one split -- the guarantee that no camera view
        # of a test sample leaks into training.
        total = len(tr) + len(va) + len(te)
        if total != len(isi):
            raise AssertionError(f"view-item expansion covered {total} of {len(isi)} items")
    else:
        tr = va = te = []

    return EvalSplit(
        seed=int(seed),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        num_samples=int(n_samples),
        train_samples=s["train"],
        val_samples=s["val"],
        test_samples=s["test"],
        train_view_items=tr,
        val_view_items=va,
        test_view_items=te,
        dataset=dataset_info,
        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    """Exercise the whole path on a synthetic dataset -- no HDF5, no GPU.

    Writes to a throwaway temp directory, never to ``prior_study_results/``: the
    synthetic fixture must not end up sitting next to the real frozen split, where a
    later reader could mistake a 137-sample toy for the study's evaluation set.
    """
    import tempfile

    import torch

    n_samples, max_views = 137, 4
    rng = np.random.RandomState(0)
    view_mask = rng.rand(n_samples, max_views) > 0.25
    view_mask[:, 0] = True  # every sample keeps at least one view
    isi = build_view_item_index(view_mask)

    split = build_eval_split(
        n_samples,
        seed=CANONICAL_SEED,
        item_sample_indices=isi,
        dataset_info={"path": "<self-test synthetic>", "num_samples": n_samples},
    )

    # 1. partition of samples
    allsamp = split.train_samples + split.val_samples + split.test_samples
    assert sorted(allsamp) == list(range(n_samples)), "sample split is not a partition"

    # 2. matches the MV trainer's call shape (random_split over a dataset object)
    n_train, n_val, n_test = split_sizes(n_samples, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    mv = torch.utils.data.random_split(
        list(range(n_samples)),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(CANONICAL_SEED),
    )
    assert sorted(int(i) for i in mv[2]) == split.test_samples, "MV split reproduction failed"

    # 3. no view of a test sample leaks into train
    train_items = set(split.train_view_items)
    for item in split.test_view_items:
        assert item not in train_items
    assert set(isi[split.test_view_items].tolist()) == set(split.test_samples)

    # 4. JSON round-trip (in a temp dir that is discarded on exit)
    with tempfile.TemporaryDirectory(prefix="eval_split_selftest_") as td:
        p = split.write(Path(td) / "eval_split_selftest.json")
        again = load_eval_split(p)
        assert again.to_dict()["samples"] == split.to_dict()["samples"]
        again.assert_compatible(n_samples, CANONICAL_SEED)

    print("self-test OK")
    print(f"  {n_samples} samples -> {n_train}/{n_val}/{n_test}")
    print(
        f"  {len(isi)} view-items -> "
        f"{len(split.train_view_items)}/{len(split.val_view_items)}/{len(split.test_view_items)}"
    )
    print("  (round-trip written to a temp dir and discarded -- nothing added to prior_study_results/)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, help="Multi-view HDF5 both arms train on (e.g. SMILySTICKS_*.h5)")
    ap.add_argument(
        "--num-samples",
        type=int,
        help="Bypass the HDF5 and split this many samples (no view-item expansion). Diagnostic use only.",
    )
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED, help=f"Split seed (canonical: {CANONICAL_SEED})")
    ap.add_argument("--train-ratio", type=float, default=CANONICAL_TRAIN_RATIO)
    ap.add_argument("--val-ratio", type=float, default=CANONICAL_VAL_RATIO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true", help="Recompute and compare against --out; exit 1 on drift")
    ap.add_argument("--force", action="store_true", help="Overwrite an existing frozen split")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Validate the logic on synthetic data (writes to a temp dir) and exit",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    if args.dataset is None and args.num_samples is None:
        ap.error("one of --dataset or --num-samples is required")

    if args.dataset is not None:
        fp, isi = probe_dataset(args.dataset)
        dataset_info = fp.__dict__.copy()
        n_samples = fp.num_samples
    else:
        isi = None
        dataset_info = {"path": None, "num_samples": args.num_samples, "note": "--num-samples override"}
        n_samples = int(args.num_samples)

    fresh = build_eval_split(
        n_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        item_sample_indices=isi,
        dataset_info=dataset_info,
    )

    if args.verify:
        stored = load_eval_split(args.out)
        try:
            stored.assert_compatible(n_samples, args.seed, args.train_ratio, args.val_ratio)
        except ValueError as e:
            print(f"VERIFY FAILED\n{e}", file=sys.stderr)
            return 1
        drift = []
        for name in ("train", "val", "test"):
            if getattr(stored, f"{name}_samples") != getattr(fresh, f"{name}_samples"):
                drift.append(f"{name} sample indices differ")
            if isi is not None and getattr(stored, f"{name}_view_items") != getattr(fresh, f"{name}_view_items"):
                drift.append(f"{name} view-item indices differ")
        stored_fp = (stored.dataset or {}).get("view_mask_sha256")
        fresh_fp = (dataset_info or {}).get("view_mask_sha256")
        if stored_fp and fresh_fp and stored_fp != fresh_fp:
            drift.append("dataset view_mask fingerprint differs -- the HDF5 changed")
        if drift:
            print("VERIFY FAILED\n  - " + "\n  - ".join(drift), file=sys.stderr)
            return 1
        print(f"VERIFY OK -- {args.out} reproduces exactly ({stored.num_samples} samples, seed {stored.seed})")
        return 0

    if args.out.exists() and not args.force:
        print(
            f"{args.out} already exists. The split is meant to be frozen once.\n"
            f"Use --verify to check it, or --force to deliberately re-freeze "
            f"(and record why in prior_study_results/PROTOCOL.md).",
            file=sys.stderr,
        )
        return 1

    path = fresh.write(args.out)
    print(f"Froze split -> {path}")
    print(f"  seed={fresh.seed}  ratios={fresh.train_ratio}/{fresh.val_ratio}/{fresh.test_ratio}")
    print(
        f"  samples:    {len(fresh.train_samples)} train / "
        f"{len(fresh.val_samples)} val / {len(fresh.test_samples)} test  (of {fresh.num_samples})"
    )
    if isi is not None:
        print(
            f"  view-items: {len(fresh.train_view_items)} train / "
            f"{len(fresh.val_view_items)} val / {len(fresh.test_view_items)} test  (of {len(isi)})"
        )
    print("\nBoth arms must now run with this seed and these ratios. See PROTOCOL.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
