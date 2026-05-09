from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from tensor_ram import FaissTensorRAM, quantize_int8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1C dummy Tensor RAM population.")
    parser.add_argument("--num-vectors", type=int, default=4096)
    parser.add_argument("--d-ram", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default="runs/ram_smoke")
    parser.add_argument("--quantize", choices=["none", "int8"], default="int8")
    parser.add_argument("--backend", choices=["numpy", "faiss"], default="numpy")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-large", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_gib = args.num_vectors * args.d_ram * 4 / (1024 ** 3)
    if raw_gib > 8 and not args.force_large:
        raise SystemExit(
            f"refusing to create a {raw_gib:.1f} GiB float32 RAM build without --force-large"
        )

    os.makedirs(args.output, exist_ok=True)
    ram = FaissTensorRAM(d_ram=args.d_ram, use_faiss=args.backend == "faiss")
    rng = np.random.default_rng(args.seed)
    remaining = args.num_vectors
    offset = 0
    while remaining:
        n = min(args.batch_size, remaining)
        vectors = rng.standard_normal((n, args.d_ram), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        ram.add(vectors)
        if args.quantize == "int8":
            q = quantize_int8(vectors)
            np.save(os.path.join(args.output, f"embeddings_int8_{offset:09d}.npy"), q.data)
            np.save(os.path.join(args.output, f"scale_{offset:09d}.npy"), q.scale)
        offset += n
        remaining -= n

    extension = ".faiss" if args.backend == "faiss" else ".npy"
    index_path = os.path.join(args.output, f"tensor_ram{extension}")
    ram.save(index_path)
    print(f"wrote {ram.ntotal} vectors to {index_path}")


if __name__ == "__main__":
    main()
