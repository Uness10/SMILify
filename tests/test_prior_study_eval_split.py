"""Tests for the frozen prior-study evaluation split (ROADMAP Phase 0).

The study's central claim rests on one structural fact: with a single seed and identical
ratios, the multi-view trainer and the single-view-from-multiview trainer put the **same
underlying frames** in the test set. These tests pin that fact against the trainers' real
inline logic, so a refactor of either split path fails here rather than silently producing
two incomparable arms months later.

What is covered:

- ``compute_sample_split`` reproduces ``train_multiview_regressor.py``'s
  ``random_split(dataset, ...)`` call **and** ``train_smil_regressor.py``'s
  ``random_split(range(n), ...)`` call, byte for byte.
- The two trainers agree with each other under one seed, and disagree under the two
  seeds currently shipped in the configs (42 vs 1234) — the exact problem Phase 0 fixes.
- View-item expansion matches ``SLEAPMultiViewDataset._build_single_view_index`` and
  leaks no camera view of a test sample into train.
- JSON round-trip, and ``assert_compatible`` catching drifted determinants.
- The shipped multi-view stick config still carries the canonical seed and ratios.

No HDF5, GPU or checkpoint is needed.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prior_study.freeze_eval_split import (  # noqa: E402
    CANONICAL_SEED,
    CANONICAL_TRAIN_RATIO,
    CANONICAL_VAL_RATIO,
    EvalSplit,
    build_eval_split,
    build_view_item_index,
    compute_sample_split,
    expand_samples_to_view_items,
    load_eval_split,
    split_sizes,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Odd counts on purpose: they exercise the int() truncation in split_sizes.
SAMPLE_COUNTS = [137, 1000, 4321]


# --------------------------------------------------------------------------- #
# reference implementations copied from the trainers
# --------------------------------------------------------------------------- #
def _mv_trainer_split(dataset_len, seed, train_ratio, val_ratio):
    """Verbatim transcription of train_multiview_regressor.py:2390-2397.

    Note it splits the *dataset object*, not a range — this is what makes the equivalence
    non-obvious and worth a test.
    """
    total_size = dataset_len
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    # Stand-in for the real Dataset. random_split only ever calls len() on it, so any
    # sequence of the right length produces the same partition — that is the property
    # under test. We read Subset.indices (dataset *indices*), which is what identifies
    # a frame; iterating the Subset would yield the real dataset's dict items.
    fake_dataset = list(range(total_size))
    train_set, val_set, test_set = torch.utils.data.random_split(
        fake_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed)
    )
    return (
        sorted(int(i) for i in train_set.indices),
        sorted(int(i) for i in val_set.indices),
        sorted(int(i) for i in test_set.indices),
    )


def _sv_trainer_split(n_samples, seed, train_ratio, val_ratio):
    """Verbatim transcription of train_smil_regressor.py:1666-1687 (sample level)."""
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    sample_train, sample_val, sample_test = torch.utils.data.random_split(
        range(n_samples),
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        sorted(int(s) for s in sample_train),
        sorted(int(s) for s in sample_val),
        sorted(int(s) for s in sample_test),
    )


def _reference_item_sample_indices(view_mask):
    """Verbatim transcription of SLEAPMultiViewDataset._build_single_view_index."""
    items = []
    for s in range(view_mask.shape[0]):
        for v in np.where(view_mask[s])[0]:
            items.append((int(s), int(v)))
    return np.array([s for (s, _v) in items], dtype=np.int64)


def _make_view_mask(n_samples, max_views=4, seed=0):
    rng = np.random.RandomState(seed)
    vm = rng.rand(n_samples, max_views) > 0.25
    vm[:, 0] = True  # every sample keeps at least one valid view
    return vm


# --------------------------------------------------------------------------- #
# split arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SAMPLE_COUNTS)
def test_split_sizes_match_trainer_truncation(n):
    """Both trainers use int() truncation, not rounding. Sizes must sum to n."""
    n_train, n_val, n_test = split_sizes(n, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    assert n_train == int(n * CANONICAL_TRAIN_RATIO)
    assert n_val == int(n * CANONICAL_VAL_RATIO)
    assert n_train + n_val + n_test == n
    assert n_test > 0


def test_split_sizes_rejects_impossible_ratios():
    with pytest.raises(ValueError):
        split_sizes(100, 0.9, 0.2)


# --------------------------------------------------------------------------- #
# equivalence with the real trainers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", SAMPLE_COUNTS)
def test_reproduces_multiview_trainer_split(n):
    ours = compute_sample_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    tr, va, te = _mv_trainer_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    assert ours["train"] == tr
    assert ours["val"] == va
    assert ours["test"] == te


@pytest.mark.parametrize("n", SAMPLE_COUNTS)
def test_reproduces_singleview_trainer_split(n):
    ours = compute_sample_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    tr, va, te = _sv_trainer_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    assert ours["train"] == tr
    assert ours["val"] == va
    assert ours["test"] == te


@pytest.mark.parametrize("n", SAMPLE_COUNTS)
def test_arms_agree_under_one_seed(n):
    """The premise of the whole study: same seed => same test frames in both arms."""
    _, _, mv_test = _mv_trainer_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    _, _, sv_test = _sv_trainer_split(n, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    assert mv_test == sv_test


def test_arms_disagree_under_the_shipped_seeds():
    """Guards the motivation for Phase 0.

    The MV stick config ships seed 42 and the SV example ships 1234. Left alone the two
    arms evaluate on different frames, and every cross-arm number is confounded. If this
    test ever starts failing, the two seeds have converged and the Phase 0 note in
    PROTOCOL.md should be revisited.
    """
    n = 1000
    _, _, mv_test = _mv_trainer_split(n, 42, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    _, _, sv_test = _sv_trainer_split(n, 1234, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)
    assert mv_test != sv_test


@pytest.mark.parametrize("n", SAMPLE_COUNTS)
def test_split_is_a_partition_and_deterministic(n):
    a = compute_sample_split(n, CANONICAL_SEED)
    b = compute_sample_split(n, CANONICAL_SEED)
    assert a == b
    assert sorted(a["train"] + a["val"] + a["test"]) == list(range(n))
    assert not (set(a["train"]) & set(a["test"]))
    assert not (set(a["val"]) & set(a["test"]))


# --------------------------------------------------------------------------- #
# view-item expansion (single-view arm)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [137, 500])
def test_view_item_index_matches_dataset_builder(n):
    vm = _make_view_mask(n)
    assert np.array_equal(build_view_item_index(vm), _reference_item_sample_indices(vm))


def test_view_item_index_rejects_wrong_rank():
    with pytest.raises(ValueError):
        build_view_item_index(np.ones(10, dtype=bool))


def test_no_view_of_a_test_sample_leaks_into_train():
    n = 500
    vm = _make_view_mask(n)
    isi = build_view_item_index(vm)
    split = build_eval_split(n, seed=CANONICAL_SEED, item_sample_indices=isi)

    # partition of items
    all_items = split.train_view_items + split.val_view_items + split.test_view_items
    assert sorted(all_items) == list(range(len(isi)))

    # every test item belongs to a test sample, and vice versa
    assert set(isi[split.test_view_items].tolist()) == set(split.test_samples)
    assert not (set(isi[split.train_view_items].tolist()) & set(split.test_samples))


def test_expansion_handles_samples_with_no_valid_views():
    """A sample whose views are all masked out owns zero items — it must not shift others."""
    vm = _make_view_mask(50)
    vm[7, :] = False
    vm[49, :] = False  # last sample: exercises the max-index bound in the expansion
    isi = build_view_item_index(vm)
    assert 7 not in set(isi.tolist())
    items = expand_samples_to_view_items([7, 49], isi)
    assert items == []
    split = build_eval_split(50, seed=CANONICAL_SEED, item_sample_indices=isi)
    all_items = split.train_view_items + split.val_view_items + split.test_view_items
    assert sorted(all_items) == list(range(len(isi)))


# --------------------------------------------------------------------------- #
# persistence + integrity guard
# --------------------------------------------------------------------------- #
def test_json_roundtrip(tmp_path):
    n = 1000
    isi = build_view_item_index(_make_view_mask(n))
    split = build_eval_split(n, seed=CANONICAL_SEED, item_sample_indices=isi)
    path = split.write(tmp_path / "eval_split.json")
    loaded = load_eval_split(path)

    assert loaded.seed == split.seed
    assert loaded.num_samples == split.num_samples
    assert loaded.test_samples == split.test_samples
    assert loaded.test_view_items == split.test_view_items
    assert abs(loaded.test_ratio - 0.10) < 1e-9

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["counts"]["test_samples"] == len(split.test_samples)


def test_load_missing_split_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="freeze_eval_split.py"):
        load_eval_split(tmp_path / "does_not_exist.json")


def test_assert_compatible_accepts_matching_determinants():
    split = build_eval_split(1000, seed=CANONICAL_SEED)
    split.assert_compatible(1000, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 999},  # dataset changed size
        {"seed": 1234},  # arm forgot to unify the seed
        {"train_ratio": 0.8},  # ratios drifted
        {"val_ratio": 0.1},
    ],
)
def test_assert_compatible_rejects_drift(kwargs):
    split = build_eval_split(1000, seed=CANONICAL_SEED)
    call = dict(
        n_samples=1000,
        seed=CANONICAL_SEED,
        train_ratio=CANONICAL_TRAIN_RATIO,
        val_ratio=CANONICAL_VAL_RATIO,
    )
    call.update(kwargs)
    with pytest.raises(ValueError, match="does not reproduce the frozen"):
        split.assert_compatible(**call)


def test_assert_compatible_catches_a_tampered_test_set():
    """Determinants match but the stored indices don't — e.g. hand-edited JSON."""
    split = build_eval_split(1000, seed=CANONICAL_SEED)
    tampered = EvalSplit.from_dict(split.to_dict())
    tampered.test_samples = tampered.test_samples[:-1]
    with pytest.raises(ValueError, match="recomputed test set differs"):
        tampered.assert_compatible(1000, CANONICAL_SEED, CANONICAL_TRAIN_RATIO, CANONICAL_VAL_RATIO)


# --------------------------------------------------------------------------- #
# config guard
# --------------------------------------------------------------------------- #
def _load_example_config(name):
    path = os.path.join(REPO_ROOT, "smal_fitter", "neuralSMIL", "configs", "examples", name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present yet")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def test_multiview_stick_config_carries_canonical_split():
    cfg = _load_example_config("multiview_sticks_UNET_optimal.json")
    assert cfg["training"]["seed"] == CANONICAL_SEED
    assert cfg["dataset"]["train_ratio"] == CANONICAL_TRAIN_RATIO
    assert cfg["dataset"]["val_ratio"] == CANONICAL_VAL_RATIO


def test_singleview_study_config_adopts_canonical_split():
    """Skips until Phase 1b creates the config; then it enforces the seed unification."""
    cfg = _load_example_config("singleview_sticks_from_mv.json")
    assert cfg["training"]["seed"] == CANONICAL_SEED, "SV study arm must use the canonical split seed"
    assert cfg["dataset"]["train_ratio"] == CANONICAL_TRAIN_RATIO
    assert cfg["dataset"]["val_ratio"] == CANONICAL_VAL_RATIO
    assert cfg["dataset"]["from_multiview"] is True
    assert cfg["dataset"]["frame_convention"] == "camera_centric"
