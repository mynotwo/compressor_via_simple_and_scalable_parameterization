"""CLI: python -m sspc {compress,decompress,bench} ..."""

from __future__ import annotations

import argparse
import bz2
import lzma
import sys
import time
import zlib
from pathlib import Path

from .codec import CodecConfig, compress, decompress
from .model import ModelConfig


def _cfg(a: argparse.Namespace) -> CodecConfig:
    return CodecConfig(
        model=ModelConfig(history=a.history, embed=a.embed, hidden=a.hidden, lr=a.lr, seed=a.seed),
        streams=a.streams,
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    d = ModelConfig()
    p.add_argument("--history", type=int, default=d.history)
    p.add_argument("--embed", type=int, default=d.embed)
    p.add_argument("--hidden", type=int, default=d.hidden)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--streams", type=int, default=CodecConfig().streams)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sspc")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compress")
    c.add_argument("src")
    c.add_argument("dst")
    _add_model_args(c)
    x = sub.add_parser("decompress")
    x.add_argument("src")
    x.add_argument("dst")
    bn = sub.add_parser("bench", help="compare against zlib/bz2/lzma and verify round trip")
    bn.add_argument("src")
    bn.add_argument("--limit", type=int, default=0, help="only use the first N bytes")
    _add_model_args(bn)
    a = ap.parse_args(argv)

    if a.cmd == "compress":
        Path(a.dst).write_bytes(compress(Path(a.src).read_bytes(), _cfg(a)))
    elif a.cmd == "decompress":
        Path(a.dst).write_bytes(decompress(Path(a.src).read_bytes()))
    else:
        data = Path(a.src).read_bytes()
        if a.limit:
            data = data[: a.limit]
        n = len(data)
        print(f"{a.src}: {n} bytes")
        for name, fn in (("zlib-9", lambda d: zlib.compress(d, 9)),
                         ("bz2-9", lambda d: bz2.compress(d, 9)),
                         ("lzma-9e", lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME))):
            out = fn(data)
            print(f"  {name:8s} {len(out):9d} B  {8 * len(out) / max(n, 1):.3f} bpb")
        t0 = time.perf_counter()
        blob = compress(data, _cfg(a))
        t1 = time.perf_counter()
        ok = decompress(blob) == data
        t2 = time.perf_counter()
        print(f"  {'sspc':8s} {len(blob):9d} B  {8 * len(blob) / max(n, 1):.3f} bpb  "
              f"enc {n / 1024 / (t1 - t0):.1f} KiB/s  dec {n / 1024 / (t2 - t1):.1f} KiB/s  "
              f"roundtrip={'ok' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
