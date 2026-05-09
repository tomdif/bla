from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuantizedEmbeddings:
    data: np.ndarray
    scale: np.ndarray
    bits: int
    original_dim: int

    def dequantize(self) -> np.ndarray:
        if self.bits == 8:
            return self.data.astype(np.float32) * self.scale[:, None]
        if self.bits == 4:
            unpacked = np.empty((self.data.shape[0], self.data.shape[1] * 2), dtype=np.uint8)
            unpacked[:, 0::2] = self.data >> 4
            unpacked[:, 1::2] = self.data & 0x0F
            signed = unpacked[:, : self.original_dim].astype(np.int16) - 8
            return signed.astype(np.float32) * self.scale[:, None]
        raise ValueError(f"unsupported quantization bits: {self.bits}")


def _row_scale(vectors: np.ndarray, denom: float) -> np.ndarray:
    max_abs = np.max(np.abs(vectors), axis=1).astype(np.float32)
    return np.maximum(max_abs / denom, 1e-8)


def quantize_int8(vectors: np.ndarray) -> QuantizedEmbeddings:
    vectors = np.asarray(vectors, dtype=np.float32)
    scale = _row_scale(vectors, 127.0)
    q = np.clip(np.round(vectors / scale[:, None]), -127, 127).astype(np.int8)
    return QuantizedEmbeddings(data=q, scale=scale, bits=8, original_dim=vectors.shape[1])


def quantize_fp4(vectors: np.ndarray) -> QuantizedEmbeddings:
    """Uniform signed 4-bit storage used as a local FP4 stand-in."""

    vectors = np.asarray(vectors, dtype=np.float32)
    scale = _row_scale(vectors, 7.0)
    q = np.clip(np.round(vectors / scale[:, None]), -8, 7).astype(np.int16) + 8
    if q.shape[1] % 2:
        q = np.pad(q, ((0, 0), (0, 1)), constant_values=8)
    packed = ((q[:, 0::2].astype(np.uint8) << 4) | q[:, 1::2].astype(np.uint8))
    return QuantizedEmbeddings(data=packed, scale=scale, bits=4, original_dim=vectors.shape[1])

