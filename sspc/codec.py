"""Online learned lossless compression: no pretrained weights, no side info.

The input is cut into ``streams`` contiguous chunks that are coded in lock
step. At step t the model predicts byte t of every chunk in one batched
forward pass, each byte is arithmetic-coded, and then the model takes one
Adam step on that batch. The decoder runs the identical sequence, so it
sees the same weights at every step. Batching across streams is the
"scalable" axis: throughput grows with the batch while the per-step Python
overhead stays constant.

Container (little-endian):
    magic b"SSPC" | version u8 | history u8 | embed u8 | hidden u16 |
    streams u16 | seed u32 | lr f64 | length u64 | crc32 u32 | payload
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from .coder import ArithmeticDecoder, ArithmeticEncoder
from .model import VOCAB, ModelConfig, OrderedMaskMLP

MAGIC = b"SSPC"
VERSION = 1
_HEADER = struct.Struct("<4sBBBHHIdQI")
FREQ_TOTAL = 1 << 16  # quantised probability scale


@dataclass(frozen=True)
class CodecConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    streams: int = 32


def quantize(prob: np.ndarray) -> np.ndarray:
    """Cumulative integer frequencies, shape (B, 257); every symbol gets >= 1."""
    freq = np.floor(prob * (FREQ_TOTAL - VOCAB)).astype(np.int64) + 1
    cum = np.zeros((prob.shape[0], VOCAB + 1), dtype=np.int64)
    np.cumsum(freq, axis=1, out=cum[:, 1:])
    return cum


def _layout(n: int, streams: int) -> tuple[int, int]:
    b = max(1, min(streams, n))
    chunk = -(-n // b) if n else 0
    return b, chunk


def _stream_lengths(n: int, b: int, chunk: int) -> np.ndarray:
    starts = np.arange(b) * chunk
    return np.clip(n - starts, 0, chunk)


def compress(data: bytes, cfg: CodecConfig = CodecConfig()) -> bytes:
    m = cfg.model
    n = len(data)
    b, chunk = _layout(n, cfg.streams)
    header = _HEADER.pack(
        MAGIC, VERSION, m.history, m.embed, m.hidden, b, m.seed, m.lr, n,
        zlib.crc32(data),
    )
    if n == 0:
        return header

    buf = np.zeros(b * chunk, dtype=np.int64)
    buf[:n] = np.frombuffer(data, dtype=np.uint8)
    grid = buf.reshape(b, chunk)  # grid[s, t] = byte t of stream s
    lengths = _stream_lengths(n, b, chunk)

    model = OrderedMaskMLP(m)
    enc = ArithmeticEncoder()
    hist = np.zeros((b, m.history), dtype=np.int64)
    for t in range(chunk):
        active = lengths > t
        sym = grid[:, t]
        prob, cache = model._forward(hist)
        cum = quantize(prob)
        for s in np.flatnonzero(active):
            x = sym[s]
            enc.encode(int(cum[s, x]), int(cum[s, x + 1]), FREQ_TOTAL)
        model.train_step(hist, sym, active.astype(np.float64), cache, prob)
        hist = np.concatenate((sym[:, None], hist[:, :-1]), axis=1)
    return header + enc.finish()


def decompress(blob: bytes) -> bytes:
    if len(blob) < _HEADER.size:
        raise ValueError("truncated header")
    magic, ver, k, d, h, b, seed, lr, n, crc = _HEADER.unpack_from(blob)
    if magic != MAGIC:
        raise ValueError("not an SSPC stream")
    if ver != VERSION:
        raise ValueError(f"unsupported version {ver}")
    if n == 0:
        return b""
    _, chunk = _layout(n, b)
    lengths = _stream_lengths(n, b, chunk)

    model = OrderedMaskMLP(ModelConfig(history=k, embed=d, hidden=h, lr=lr, seed=seed))
    dec = ArithmeticDecoder(blob[_HEADER.size:])
    grid = np.zeros((b, chunk), dtype=np.int64)
    hist = np.zeros((b, k), dtype=np.int64)
    for t in range(chunk):
        active = lengths > t
        prob, cache = model._forward(hist)
        cum = quantize(prob)
        sym = np.zeros(b, dtype=np.int64)
        for s in np.flatnonzero(active):
            row = cum[s]
            target = dec.target(FREQ_TOTAL)
            x = int(np.searchsorted(row, target, side="right")) - 1
            dec.consume(int(row[x]), int(row[x + 1]), FREQ_TOTAL)
            sym[s] = x
        grid[:, t] = sym
        model.train_step(hist, sym, active.astype(np.float64), cache, prob)
        hist = np.concatenate((sym[:, None], hist[:, :-1]), axis=1)

    out = grid.reshape(-1)[:n].astype(np.uint8).tobytes()
    if zlib.crc32(out) != crc:
        raise ValueError("CRC mismatch: corrupt stream or non-reproducible float math")
    return out
