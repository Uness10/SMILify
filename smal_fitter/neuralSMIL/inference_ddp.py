"""Distributed-inference plumbing shared by the single-view and multi-view entrypoints.

``run_multiview_inference.py`` has run multi-GPU since it was written; when
``run_singleview_inference.py`` gained a dataset mode (issue #100) it needed the
same machinery. Rather than copy it — the duplication that caused the render
paths to drift in the first place — the generic parts live here:

* the IPv4-only ``getaddrinfo`` patch HPC nodes without full IPv6 need,
* process-group setup / teardown with an explicit IPv4 ``TCPStore``,
* the striped rank -> dataset-index assignment,
* gathering per-rank predictions through shared temp storage,
* gathering per-rank rendered frames through shared temp storage.

Nothing here knows about single- vs multi-view: frames are handed over as named
*streams* (``"mv"``, ``"sv_view0"``, ``"sv"``, ...) so either caller can describe
whatever set of videos it writes.

NOTE ON IMPORT ORDER: :func:`force_ipv4_getaddrinfo` must run before any name
resolution happens. Entrypoints apply the patch at the very top of their own
module (before ``import torch``) and this module is import-safe either way.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import socket
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist


# --------------------------------------------------------------------------- #
# IPv4 forcing
# --------------------------------------------------------------------------- #

_original_getaddrinfo = socket.getaddrinfo
_ipv4_patch_applied = False


def _getaddrinfo_ipv4_only(*args, **kwargs):
    """Force ``getaddrinfo`` to return only IPv4 results."""
    responses = _original_getaddrinfo(*args, **kwargs)
    ipv4_responses = [r for r in responses if r[0] == socket.AF_INET]
    # If we have IPv4 results, use them; otherwise fall back to the original.
    return ipv4_responses if ipv4_responses else responses


def force_ipv4_getaddrinfo() -> None:
    """Patch ``socket.getaddrinfo`` to hide IPv6 results. Idempotent.

    Prevents "Address family not supported by protocol" (errno 97) on HPC
    systems without full IPv6 support, where the default resolution of
    ``MASTER_ADDR`` returns an AAAA record NCCL/gloo then fail to bind.
    """
    global _ipv4_patch_applied
    if _ipv4_patch_applied:
        return
    socket.getaddrinfo = _getaddrinfo_ipv4_only
    _ipv4_patch_applied = True


# --------------------------------------------------------------------------- #
# Process group
# --------------------------------------------------------------------------- #


def is_torchrun_launched() -> bool:
    """True when launched via ``torchrun`` / ``torch.distributed.launch``.

    Those launchers set ``RANK``, ``LOCAL_RANK`` and ``WORLD_SIZE``; ``mp.spawn``
    does not, which is how the entrypoints tell the two launch modes apart.
    """
    return all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])


def resolve_dist_timeout(timeout_s: Optional[int] = None) -> int:
    """Resolve the process-group timeout in seconds.

    Defaults to ``SMILIFY_DIST_TIMEOUT_S``, else 14400 (4 h). Inference has long
    rank-0-only phases BETWEEN collectives (gathering ~100k predictions from temp
    pickles, writing animation exports, merging videos) during which the other
    ranks sit at a ``dist.barrier()``; with the old hard-coded 1800 s the NCCL
    watchdog killed those ranks on large datasets (ALLREDUCE timeout at the
    barrier) and took the whole job down after inference had already succeeded.
    """
    if timeout_s is not None:
        return int(timeout_s)
    raw = os.environ.get("SMILIFY_DIST_TIMEOUT_S", "14400")
    try:
        return int(raw)
    except ValueError:
        print(f"[resolve_dist_timeout] invalid SMILIFY_DIST_TIMEOUT_S={raw!r}; using default 14400 s")
        return 14400


def setup_ddp(
    rank: int,
    world_size: int,
    port: str = "12345",
    local_rank: Optional[int] = None,
    timeout_s: Optional[int] = None,
) -> None:
    """Initialize the DDP environment with a robust IPv4-only TCP store.

    Args:
        rank: Current process rank (global rank across all nodes)
        world_size: Total number of processes
        port: Master port (ignored when ``MASTER_PORT`` is set)
        local_rank: Local rank within the node (for GPU assignment); defaults to *rank*
        timeout_s: Process-group / store timeout in seconds; see
            :func:`resolve_dist_timeout` for the default and why it is large.
    """
    timeout_s = resolve_dist_timeout(timeout_s)
    dist_timeout = timedelta(seconds=timeout_s)

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", port or "12345"))

    # Validate that master_addr is an IPv4 address (not a hostname)
    ipv4_pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
    if not re.match(ipv4_pattern, master_addr):
        print(f"WARNING: MASTER_ADDR '{master_addr}' is not an IPv4 address!")
        print("  Attempting to resolve to IPv4...")
        try:
            result = socket.getaddrinfo(master_addr, master_port, socket.AF_INET, socket.SOCK_STREAM)
            if result:
                master_addr = result[0][4][0]
                print(f"  Resolved to: {master_addr}")
            else:
                print(f"  ERROR: Could not resolve {master_addr} to IPv4!")
        except Exception as e:
            print(f"  ERROR resolving hostname: {e}")

    # Use local_rank for GPU assignment (important for multi-node setups).
    # Do this BEFORE init_process_group so NCCL binds to the correct GPU.
    gpu_rank = local_rank if local_rank is not None else rank

    if rank == 0:
        print(f"CUDA devices available: {torch.cuda.device_count()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    torch.cuda.set_device(gpu_rank)

    if dist.is_initialized():
        return

    print(
        f"[Rank {rank}] Initializing distributed: WORLD_SIZE={world_size}, "
        f"LOCAL_RANK/GPU={gpu_rank}, MASTER={master_addr}:{master_port}"
    )

    try:
        # Explicit TCPStore with an IPv4 address, bypassing the env:// default
        # whose hostname resolution can return IPv6.
        store = dist.TCPStore(
            host_name=master_addr,
            port=master_port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=dist_timeout,
            use_libuv=False,  # Disable libuv to avoid potential IPv6 issues
        )
        dist.init_process_group(backend="nccl", store=store, rank=rank, world_size=world_size, timeout=dist_timeout)
        print(f"[Rank {rank}] Successfully initialized NCCL process group (timeout={timeout_s}s)")

    except Exception as e:
        print(f"Error initializing process group with NCCL + TCPStore: {e}")
        print(f"  MASTER_ADDR: {master_addr}")
        print(f"  MASTER_PORT: {master_port}")
        print(f"  RANK: {rank}, WORLD_SIZE: {world_size}")
        print(f"  LOCAL_RANK: {local_rank}, GPU_RANK: {gpu_rank}")

        print("Attempting fallback to gloo backend with TCPStore...")
        try:
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port + 1,  # Use a different port for gloo
                world_size=world_size,
                is_master=(rank == 0),
                timeout=dist_timeout,
                use_libuv=False,
            )
            dist.init_process_group(
                backend="gloo", store=store, rank=rank, world_size=world_size, timeout=dist_timeout
            )
            print(f"[Rank {rank}] Successfully initialized with gloo backend!")
        except Exception as e2:
            print(f"Gloo fallback also failed: {e2}")
            raise e  # Re-raise the original NCCL error


def cleanup_ddp() -> None:
    """Tear down the DDP environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """``dist.barrier()`` that is a no-op outside a process group."""
    if dist.is_initialized():
        dist.barrier()


# --------------------------------------------------------------------------- #
# Work assignment
# --------------------------------------------------------------------------- #


def compute_rank_indices(
    dataset_size: int,
    rank: int,
    world_size: int,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> List[int]:
    """Return the dataset indices in ``[start_idx, end_idx)`` assigned to *rank*.

    With ``world_size > 1`` the range is striped across ranks (rank 0 takes
    ``start_idx``, ``start_idx + world_size``, ...). Striping — rather than
    contiguous chunks — keeps each rank's slice spread over the whole clip, so a
    rank that dies does not remove one continuous span of frames and per-rank
    runtimes stay balanced when sample cost varies along the sequence.
    """
    if end_idx is None:
        end_idx = dataset_size
    end_idx = min(end_idx, dataset_size)
    start_idx = max(0, start_idx)

    if world_size > 1:
        return list(range(start_idx + rank, end_idx, world_size))
    return list(range(start_idx, end_idx))


# --------------------------------------------------------------------------- #
# Prediction gathering
# --------------------------------------------------------------------------- #


def write_predictions_to_temp(
    raw_predictions: List[Tuple[int, dict]],
    temp_dir: Path,
    rank: int,
) -> None:
    """Pickle this rank's raw predictions to shared temp storage."""
    rank_dir = temp_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    pred_path = rank_dir / "predictions.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(raw_predictions, f)
    print(f"[Rank {rank}] Wrote {len(raw_predictions)} predictions to {pred_path}")


def load_all_predictions_from_temp(
    temp_dir: Path,
    world_size: int,
) -> List[Tuple[int, dict]]:
    """Load and merge every rank's prediction pickle, sorted by global index."""
    all_predictions: List[Tuple[int, dict]] = []
    for rank_idx in range(world_size):
        pred_path = temp_dir / f"rank_{rank_idx}" / "predictions.pkl"
        if pred_path.exists():
            with open(pred_path, "rb") as f:
                all_predictions.extend(pickle.load(f))
    all_predictions.sort(key=lambda x: x[0])
    print(f"Loaded {len(all_predictions)} total predictions from {world_size} ranks")
    return all_predictions


def gather_predictions(
    raw_predictions: List[Tuple[int, dict]],
    rank: int,
    world_size: int,
    temp_dir: Path,
    all_ranks: bool = False,
    cleanup: bool = True,
) -> Optional[List[Tuple[int, dict]]]:
    """Gather per-rank predictions through *temp_dir*, sorted by global index.

    Predictions travel via pickles on shared storage rather than an NCCL
    all-gather: the payload is large, ragged and CPU-resident, and gathering
    ~100k of them through the collective is both slow and a memory spike on
    rank 0.

    Args:
        all_ranks: when True every rank receives the full list (needed so each
            rank can build the same temporally-ordered sequence for smoothing);
            when False only rank 0 does and the others get ``None``.
        cleanup: remove the per-rank ``predictions.pkl`` files afterwards. The
            *directory* is left in place, so a caller that reuses ``temp_dir``
            for frame storage later does not have to recreate it.

    Returns:
        The sorted ``(global_idx, params)`` list, or ``None`` on non-zero ranks
        when ``all_ranks`` is False.
    """
    if world_size <= 1:
        return sorted(raw_predictions, key=lambda x: x[0])

    if rank == 0:
        temp_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    write_predictions_to_temp(raw_predictions, temp_dir, rank)
    barrier()

    result: Optional[List[Tuple[int, dict]]] = None
    if all_ranks or rank == 0:
        result = load_all_predictions_from_temp(temp_dir, world_size)

    # Everyone must be done reading before anything is deleted.
    barrier()
    if cleanup and rank == 0:
        for rank_idx in range(world_size):
            pred_path = temp_dir / f"rank_{rank_idx}" / "predictions.pkl"
            if pred_path.exists():
                pred_path.unlink()
    barrier()

    return result


# --------------------------------------------------------------------------- #
# Frame gathering
# --------------------------------------------------------------------------- #

FrameStream = Tuple[List[np.ndarray], List[int]]
"""A named video stream: ``(frames, global_indices)``, index-aligned."""


def write_frame_streams_to_temp(
    streams: Dict[str, FrameStream],
    temp_dir: Path,
    rank: int,
) -> Path:
    """Write this rank's rendered frames to shared temp storage as PNGs.

    Frames go to disk rather than through an all-gather because a full clip of
    rendered collages does not fit in memory on one rank.

    Args:
        streams: ``{stream_name: (frames, global_indices)}``. Stream names become
            filename prefixes, so keep them path-safe (``"mv"``,
            ``"sv_view0"``, ``"sv"``).

    Returns:
        The per-rank directory holding the PNGs and the manifest.
    """
    import cv2

    rank_dir = temp_dir / f"rank_{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, List[Tuple[int, str]]] = {}
    total = 0
    for stream_name, (frames, indices) in streams.items():
        entries: List[Tuple[int, str]] = []
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            frame_path = rank_dir / f"{stream_name}_{i:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            entries.append((idx, str(frame_path)))
        manifest[stream_name] = entries
        total += len(entries)

    with open(rank_dir / "frame_manifest.pkl", "wb") as f:
        pickle.dump(manifest, f)

    print(f"[Rank {rank}] Wrote {total} frames across {len(streams)} stream(s) to {rank_dir}")
    return rank_dir


def merge_frame_streams_from_temp(
    temp_dir: Path,
    world_size: int,
    stream_names: List[str],
) -> Dict[str, List[Tuple[int, str]]]:
    """Read every rank's manifest and return each stream ordered by global index.

    Called on rank 0 only. Sorting by the original dataset index is what undoes
    the striping applied by :func:`compute_rank_indices`.
    """
    merged: Dict[str, List[Tuple[int, str]]] = {name: [] for name in stream_names}

    for rank_idx in range(world_size):
        manifest_path = temp_dir / f"rank_{rank_idx}" / "frame_manifest.pkl"
        if not manifest_path.exists():
            continue
        with open(manifest_path, "rb") as f:
            manifest = pickle.load(f)
        for stream_name, entries in manifest.items():
            merged.setdefault(stream_name, []).extend(entries)

    for stream_name in merged:
        merged[stream_name].sort(key=lambda x: x[0])
    return merged


def write_video_from_manifest(
    entries: List[Tuple[int, str]],
    out_path: Path,
    fps: float,
    size: Tuple[int, int],
    fourcc: str = "mp4v",
) -> int:
    """Write the PNGs named by *entries* (already ordered) into a video file."""
    import cv2

    if not entries:
        return 0

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {out_path} at {size[0]}x{size[1]}")
    written = 0
    try:
        for _, frame_path in entries:
            frame = cv2.imread(frame_path)
            if frame is not None:
                writer.write(frame)
                written += 1
    finally:
        writer.release()
    return written


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove *temp_dir* if it exists (best effort)."""
    shutil.rmtree(temp_dir, ignore_errors=True)


def resolve_launch(rank: int, world_size: int) -> Tuple[int, int, int]:
    """Return ``(rank, world_size, gpu_rank)`` for the active launch mode.

    Under ``torchrun`` / SLURM the environment is authoritative and the spawn
    arguments are ignored; under ``mp.spawn`` the local rank equals the rank.
    """
    if is_torchrun_launched():
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        return rank, world_size, local_rank
    return rank, world_size, rank


def validate_num_gpus(num_gpus: int) -> None:
    """Fail fast when more GPUs are requested than the node has."""
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU processing requested but CUDA is not available")
    available = torch.cuda.device_count()
    if num_gpus > available:
        raise RuntimeError(f"Requested {num_gpus} GPUs but only {available} available")
