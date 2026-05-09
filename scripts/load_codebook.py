"""Load a pretrained embedding table into the DCA's lexical codebook.

Accepts .npy (numpy array of shape [vocab, d_core]) or .pt (torch tensor).
The codebook is L2-normalized by default so argmax decoding behaves as
cosine similarity. The result is written next to the input as
`<name>.codebook.pt` ready to load via DeterministicLexicalDecoder.from_embeddings.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

from system2_dca import DeterministicLexicalDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".npy or .pt embedding tensor")
    parser.add_argument("--d-core", type=int, required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


def load_tensor(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        import numpy as np

        arr = np.load(path)
        return torch.from_numpy(arr)
    if path.endswith(".pt") or path.endswith(".pth"):
        return torch.load(path, map_location="cpu")
    raise ValueError(f"unsupported file extension: {path}")


def main() -> None:
    args = parse_args()
    weight = load_tensor(args.input).to(dtype=torch.float32)
    if weight.ndim != 2:
        raise SystemExit(f"expected 2D embedding tensor, got {tuple(weight.shape)}")
    if weight.shape[1] != args.d_core:
        raise SystemExit(
            f"embedding dim {weight.shape[1]} does not match --d-core {args.d_core}"
        )
    decoder = DeterministicLexicalDecoder.from_embeddings(
        weight, normalize_codebook=not args.no_normalize
    )
    output = args.output or args.input.rsplit(".", 1)[0] + ".codebook.pt"
    torch.save({"codebook": decoder.codebook.detach().cpu(), "vocab_size": weight.shape[0], "d_core": weight.shape[1]}, output)
    print(f"wrote codebook of shape {tuple(decoder.codebook.shape)} to {output}")


if __name__ == "__main__":
    main()
