#!/usr/bin/env python3
"""
Extract a raw video clip from the SMILySTICKS HDF5 dataset.

The dataset stores JPEG-encoded frames per view. Frames are ALREADY
centre-cropped to 512x512 (metadata.crop_mode == 'centred'), so a clip
extracted here can be fed straight to run_singleview_inference with
--crop_mode centred: the crop is then a no-op and only the 512->224
resize applies. That keeps inference consistent with training.

Usage
-----
  # see what's in the file
  python extract_clip.py --h5 SMILySTICKS_centred_reprojected_FIXED.h5 --list

  # extract 300 consecutive frames of one session, camera A
  python extract_clip.py --h5 SMILySTICKS_centred_reprojected_FIXED.h5 \
      --session <name> --camera A --start 0 --length 300 \
      --out prior_study_results/renders/clip_A.mp4
"""
import argparse
import h5py
import numpy as np
import cv2


def as_str(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--list", action="store_true", help="print sessions/cameras and exit")
    ap.add_argument("--session", default=None, help="session name (default: first)")
    ap.add_argument("--camera", default=None, help="camera name, e.g. A (default: first)")
    ap.add_argument("--start", type=int, default=0, help="starting frame_idx")
    ap.add_argument("--length", type=int, default=300, help="number of frames")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", default="clip.mp4")
    ap.add_argument("--also-frames", default=None,
                    help="optional dir; also write the frames as PNGs")
    a = ap.parse_args()

    with h5py.File(a.h5, "r") as f:
        sessions = np.array([as_str(s) for s in f["auxiliary/session_name"][:]])
        cameras = np.array([as_str(s) for s in f["auxiliary/camera_names"][:]])
        frame_idx = f["auxiliary/frame_idx"][:]
        order = f["metadata"].attrs.get("canonical_camera_order", '["A","B","C","D","E"]')
        if isinstance(order, (bytes, bytearray)):
            order = order.decode()
        import json
        cam_order = json.loads(order) if isinstance(order, str) else list(order)

        if a.list:
            print("canonical camera order:", cam_order)
            for s in np.unique(sessions):
                m = sessions == s
                print(f"  session {s!r}: {m.sum()} samples, "
                      f"frame_idx {frame_idx[m].min()}..{frame_idx[m].max()}, "
                      f"cameras {sorted(set(cameras[m]))}")
            return

        session = a.session or str(np.unique(sessions)[0])
        camera = a.camera or cam_order[0]
        if camera not in cam_order:
            raise SystemExit(f"camera {camera!r} not in {cam_order}")
        view = cam_order.index(camera)

        # Rows for this session, in frame order, where this view actually exists.
        mask = (sessions == session)
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(frame_idx[rows], kind="stable")]

        view_mask = f["multiview_images/view_mask"]
        keep = [r for r in rows if view_mask[r, view]]
        keep = [r for r in keep if frame_idx[r] >= a.start][: a.length]
        if not keep:
            raise SystemExit(
                f"no frames for session={session!r} camera={camera!r} start={a.start}"
            )

        # Warn on gaps: a jump in frame_idx means the clip is not continuous,
        # which would corrupt any gait-cycle analysis downstream.
        fi = frame_idx[keep]
        gaps = np.flatnonzero(np.diff(fi) != 1)
        if len(gaps):
            print(f"WARNING: {len(gaps)} discontinuity/ies in frame_idx "
                  f"(first at frame {fi[gaps[0]]}). The clip is NOT continuous.")

        jpegs = f[f"multiview_images/image_jpeg_view_{view}"]
        writer = None
        for n, r in enumerate(keep):
            buf = np.frombuffer(bytes(jpegs[r]), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # BGR
            if img is None:
                print(f"skip: undecodable frame at row {r}")
                continue
            if writer is None:
                h, w = img.shape[:2]
                writer = cv2.VideoWriter(
                    a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h)
                )
                print(f"writing {a.out}  {w}x{h} @ {a.fps} fps")
            writer.write(img)
            if a.also_frames:
                import os
                os.makedirs(a.also_frames, exist_ok=True)
                cv2.imwrite(f"{a.also_frames}/frame_{fi[n]:06d}.png", img)
        writer.release()
        print(f"done: session={session!r} camera={camera} "
              f"frames {fi[0]}..{fi[-1]} ({len(keep)})")


if __name__ == "__main__":
    main()
