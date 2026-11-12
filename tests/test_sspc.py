import os
import random

import numpy as np
import pytest

from sspc import CodecConfig, ModelConfig, OrderedMaskMLP, compress, decompress
from sspc.coder import ArithmeticDecoder, ArithmeticEncoder

SMALL = CodecConfig(model=ModelConfig(history=4, embed=4, hidden=16), streams=8)


def test_arithmetic_coder_roundtrip_random_intervals():
    rng = random.Random(1)
    total = 1000
    syms = []
    enc = ArithmeticEncoder()
    for _ in range(5000):
        lo = rng.randrange(total - 1)
        hi = rng.randrange(lo + 1, total + 1) if rng.random() < 0.5 else lo + 1
        syms.append((lo, hi))
        enc.encode(lo, hi, total)
    dec = ArithmeticDecoder(enc.finish())
    for lo, hi in syms:
        t = dec.target(total)
        assert lo <= t < hi
        dec.consume(lo, hi, total)


@pytest.mark.parametrize(
    "data",
    [b"", b"a", b"\x00\xff" * 3, bytes(range(256)), os.urandom(500),
     b"the quick brown fox jumps over the lazy dog. " * 40],
)
def test_codec_roundtrip(data):
    assert decompress(compress(data, SMALL)) == data


def test_more_streams_than_bytes():
    data = b"xyz"
    assert decompress(compress(data, CodecConfig(model=SMALL.model, streams=64))) == data


def test_learns_redundant_input():
    data = b"abcdefgh" * 400
    blob = compress(data, CodecConfig(model=ModelConfig(history=8, embed=8, hidden=32, lr=1e-2), streams=8))
    assert len(blob) < len(data) / 4


def test_corruption_detected():
    blob = bytearray(compress(b"hello world " * 50, SMALL))
    blob[-5] ^= 0xFF
    with pytest.raises(ValueError):
        decompress(bytes(blob))


def test_mask_monotone_and_gradients_match_finite_differences():
    m = OrderedMaskMLP(ModelConfig(history=5, embed=3, hidden=7, seed=3))
    rng = np.random.default_rng(0)
    m.params["theta"] = rng.normal(0, 1, 4)
    w = m.mask()
    assert w[0] == 1.0 and np.all(np.diff(w) < 0)

    ctx = rng.integers(0, 256, (6, 5))
    tgt = rng.integers(0, 256, 6)
    act = np.array([1, 1, 1, 0, 1, 1], dtype=float)
    _, g = m.gradients(ctx, tgt, act)
    eps = 1e-6
    for name in ("theta", "W1", "b1", "W2", "E"):
        p = m.params[name]
        for idx in [tuple(rng.integers(0, s) for s in p.shape) for _ in range(4)]:
            if name == "E":
                idx = (int(ctx[0, 1]),) + idx[1:]
            old = p[idx]
            p[idx] = old + eps
            lp, _ = m.gradients(ctx, tgt, act)
            p[idx] = old - eps
            lm, _ = m.gradients(ctx, tgt, act)
            p[idx] = old
            assert g[name][idx] == pytest.approx((lp - lm) / (2 * eps), rel=1e-4, abs=1e-8)
