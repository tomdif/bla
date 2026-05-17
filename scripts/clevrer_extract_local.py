"""Extract a CLEVRER subset into a compact local .npz cache.

For each video, decodes the MP4 + reads the matching scene-annotation
JSON. Output per episode:
  video           [T, H, W, 3] uint8
  image_positions [E, T, 2]    float32 in [0, 1] (mask centroid)
  visibility      [E, T]       uint8 (1 if detected at this frame)
  color_idx       [E]          int32 (index into clevrer_data.COLORS)
  material_idx    [E]          int32
  shape_idx       [E]          int32
  num_instances   scalar       int32
  video_name      scalar       str
  collisions      object       list[(frame_idx, obj_a_id, obj_b_id)]
  in_outs         object       list[(frame_idx, obj_id, type)]  # type=0 in, 1 out

Then a `manifest.json` index. Usage:

    python scripts/clevrer_extract_local.py \\
        --videos /workspace/clevrer/videos/train \\
        --annotations /workspace/clevrer/annotations/processed_proposals \\
        --out /workspace/clevrer_local/train \\
        --max-episodes 1000 --frame-stride 4 --frame-size 128
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def decode_video(path: str, frame_stride: int, frame_size: int) -> np.ndarray:
    """Read MP4 → [T_strided, frame_size, frame_size, 3] uint8 via PyAV."""
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i % frame_stride != 0:
            continue
        # Convert + resize to target size.
        img = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
        if img.shape[0] != frame_size or img.shape[1] != frame_size:
            # Resize via PIL (np→PIL→np); cheap for small frames.
            from PIL import Image
            pil = Image.fromarray(img)
            pil = pil.resize((frame_size, frame_size), Image.BILINEAR)
            img = np.asarray(pil)
        frames.append(img)
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)  # [T, H, W, 3] uint8


def extract_one(video_path: Path, ann_path: Path, frame_stride: int,
                frame_size: int) -> dict:
    """Read video + annotation for one CLEVRER episode."""
    from system1_jepa.clevrer_data import (
        COLORS, MATERIALS, SHAPES, rle_to_bbox, _match_detection_to_gt,
    )

    with open(ann_path) as f:
        ann = json.load(f)

    gt_objects = ann["ground_truth"]["objects"]
    E = len(gt_objects)
    # Object metadata (attribute indices into vocab).
    color_idx = np.array([COLORS.index(o["color"]) for o in gt_objects], dtype=np.int32)
    material_idx = np.array([MATERIALS.index(o["material"]) for o in gt_objects], dtype=np.int32)
    shape_idx = np.array([SHAPES.index(o["shape"]) for o in gt_objects], dtype=np.int32)

    # Per-frame positions + visibility (E × T).
    # CLEVRER has 128 frames per video; we stride to ~32 frames at stride=4.
    full_frames = ann["frames"]
    total_T = len(full_frames)
    T = (total_T + frame_stride - 1) // frame_stride

    pos = np.zeros((E, T, 2), dtype=np.float32)
    vis = np.zeros((E, T), dtype=np.uint8)
    for t_strided in range(T):
        t = t_strided * frame_stride
        if t >= total_T: break
        frame_data = full_frames[t]
        for det in frame_data["objects"]:
            obj_id = _match_detection_to_gt(
                (det["color"], det["material"], det["shape"]), gt_objects
            )
            if obj_id < 0 or obj_id >= E:
                continue
            mask = det["mask"]
            centroid = rle_to_bbox(mask["size"], mask["counts"])
            if centroid is None:
                continue
            pos[obj_id, t_strided] = centroid
            vis[obj_id, t_strided] = 1

    # Collision + in_out events, stride-aligned.
    collisions_strided = []
    for ev in ann["ground_truth"].get("collisions", []):
        f_strided = ev["frame"] // frame_stride
        if f_strided < T:
            collisions_strided.append((f_strided, ev["object"][0], ev["object"][1]))

    in_outs_strided = []
    for ev in ann["ground_truth"].get("in_outs", []):
        f_strided = ev["frame"] // frame_stride
        if f_strided < T:
            type_idx = 0 if ev["type"] == "in" else 1
            in_outs_strided.append((f_strided, ev["object"], type_idx))

    # Decode video.
    video = decode_video(str(video_path), frame_stride, frame_size)
    if video.shape[0] != T:
        # Truncate annotation T to match actual video frames.
        T_actual = min(video.shape[0], T)
        pos = pos[:, :T_actual]
        vis = vis[:, :T_actual]
        video = video[:T_actual]

    return {
        "video": video,
        "image_positions": pos,
        "visibility": vis,
        "color_idx": color_idx,
        "material_idx": material_idx,
        "shape_idx": shape_idx,
        "num_instances": np.int32(E),
        "video_name": str(np.array(video_path.stem)),
        "collisions": np.array(collisions_strided, dtype=object),
        "in_outs": np.array(in_outs_strided, dtype=object),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="dir of CLEVRER MP4s")
    p.add_argument("--annotations", required=True, help="dir of processed_proposals/sim_*.json")
    p.add_argument("--out", required=True)
    p.add_argument("--max-episodes", type=int, default=200)
    p.add_argument("--frame-stride", type=int, default=4)
    p.add_argument("--frame-size", type=int, default=128)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Add the repo root to sys.path so clevrer_data imports work.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Find all annotation files; for each, look for the matching video.
    ann_files = sorted(Path(args.annotations).glob("sim_*.json"))
    if args.max_episodes > 0:
        ann_files = ann_files[: args.max_episodes]
    print(f"Found {len(ann_files)} annotation files; extracting first {len(ann_files)}", flush=True)

    manifest = []
    for i, ann_path in enumerate(ann_files):
        # Video file convention: video_{NNNNN}.mp4 in shards.
        video_idx = int(ann_path.stem.split("_")[1])  # e.g. sim_02780 → 2780
        # CLEVRER organizes videos into shards of 1000 each: video_NNNNN.mp4 in
        # subdirs video_00000-01000/, video_01000-02000/, etc.
        shard_start = (video_idx // 1000) * 1000
        shard_end = shard_start + 1000
        shard_dir = Path(args.videos) / f"video_{shard_start:05d}-{shard_end:05d}"
        video_path = shard_dir / f"video_{video_idx:05d}.mp4"
        if not video_path.exists():
            print(f"  [skip] missing {video_path}", flush=True)
            continue

        try:
            ep = extract_one(video_path, ann_path, args.frame_stride, args.frame_size)
        except Exception as e:
            print(f"  [error] {ann_path.name}: {e}", flush=True)
            continue

        ep_path = out / f"ep_{i:05d}.npz"
        np.savez_compressed(ep_path, **ep)
        manifest.append({
            "ep_id": i,
            "file": ep_path.name,
            "video_name": str(ep["video_name"]),
            "num_instances": int(ep["num_instances"]),
            "T": int(ep["video"].shape[0]),
        })
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(ann_files)}] cached {ep['video_name']}", flush=True)

    with open(out / "manifest.json", "w") as f:
        json.dump({
            "n_episodes": len(manifest),
            "frame_stride": args.frame_stride,
            "frame_size": args.frame_size,
            "episodes": manifest,
        }, f, indent=2)

    total_mb = sum(os.path.getsize(out / m["file"]) for m in manifest) / 1e6
    print(f"\nDone. {len(manifest)} episodes, {total_mb:.1f} MB total.", flush=True)


if __name__ == "__main__":
    main()
