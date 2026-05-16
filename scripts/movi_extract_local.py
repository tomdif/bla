"""One-time extractor: stream MOVi-A (or other MOVi variants) from public GCS
via TFDS, drop unused fields, and save a compact local cache.

For our Phase 7 evaluation we only need:
  - video frames [T, H, W, 3] uint8
  - per-instance image_positions [E, T, 2] float32 (projected 2D pixel coords)
  - per-instance visibility [E, T] uint16 (>0 → visible)
  - per-instance color/shape/material/size labels (int)
  - num_instances (int)

We drop:
  - depth, normal, object_coordinates, forward_flow, backward_flow,
    segmentations, bboxes, bboxes_3d, camera intrinsics, collisions
  - 3D positions and quaternions (we project to 2D anyway)

This shrinks each episode from ~80MB to ~5MB. The cache is .npz per
episode + a manifest.json so the loader can mmap and shuffle by file.

Usage:
    python scripts/movi_extract_local.py --dataset movi_a --split validation \\
        --out /workspace/movi_a_local/validation --max-episodes 200
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="movi_a",
                        help="Which MOVi variant: movi_a..movi_f")
    parser.add_argument("--split", default="validation",
                        choices=["train", "validation"])
    parser.add_argument("--out", required=True,
                        help="Output directory for .npz files + manifest")
    parser.add_argument("--max-episodes", type=int, default=200,
                        help="Cap episode count (0 = all)")
    parser.add_argument("--frame-size", type=int, default=0,
                        help="If >0, resize frames to this square size on save")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf  # noqa: E402
    import tensorflow_datasets as tfds  # noqa: E402

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split_str = f"{args.split}[:{args.max_episodes}]" if args.max_episodes else args.split
    ds = tfds.load(f"kubric:{args.dataset}", split=split_str)

    manifest = []
    for i, ex in enumerate(ds):
        video = ex["video"].numpy()                              # [T, H, W, 3] uint8
        inst = ex["instances"]
        image_positions = inst["image_positions"].numpy()        # [E, T, 2] float
        visibility = inst["visibility"].numpy()                  # [E, T] uint16
        color_label = inst["color_label"].numpy().astype(np.int32)   # [E]
        shape_label = inst["shape_label"].numpy().astype(np.int32)   # [E]
        material_label = inst["material_label"].numpy().astype(np.int32)  # [E]
        size_label = inst["size_label"].numpy().astype(np.int32)     # [E]
        num_instances = int(ex["metadata"]["num_instances"].numpy())
        video_name = ex["metadata"]["video_name"].numpy().decode("utf-8")

        if args.frame_size > 0 and args.frame_size != video.shape[1]:
            # Resize via tf.image (bilinear) to preserve smoothness for slot binding
            import tensorflow as tf
            v = tf.convert_to_tensor(video)
            v = tf.image.resize(v, [args.frame_size, args.frame_size], method="bilinear")
            new_video = tf.cast(tf.clip_by_value(v, 0, 255), tf.uint8).numpy()
            scale = args.frame_size / video.shape[1]
            new_image_positions = image_positions * scale
            video = new_video
            image_positions = new_image_positions

        ep_path = out / f"ep_{i:05d}.npz"
        np.savez_compressed(
            ep_path,
            video=video,
            image_positions=image_positions.astype(np.float32),
            visibility=(visibility > 0).astype(np.uint8),
            color_label=color_label,
            shape_label=shape_label,
            material_label=material_label,
            size_label=size_label,
            num_instances=np.int32(num_instances),
            video_name=np.array(video_name),
        )
        manifest.append({
            "ep_id": i,
            "file": ep_path.name,
            "num_instances": num_instances,
            "video_name": video_name,
            "T": int(video.shape[0]),
            "H": int(video.shape[1]),
            "W": int(video.shape[2]),
        })
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{args.max_episodes or '?'}] cached {video_name}", flush=True)
        if args.max_episodes and i + 1 >= args.max_episodes:
            break

    with open(out / "manifest.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "n_episodes": len(manifest),
            "frame_size": args.frame_size,
            "episodes": manifest,
        }, f, indent=2)

    total_mb = sum(os.path.getsize(out / m["file"]) for m in manifest) / 1e6
    print(f"\nDone. {len(manifest)} episodes, {total_mb:.1f} MB total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
