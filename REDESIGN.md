# Redesign notes

## Starting point

The repo held only a README pointing at the MM'22 paper and a license; the
paper's implementation was not included.

## Purpose

A small, readable, runnable reference implementation of the paper's core
idea: learned general-purpose lossless compression where history importance
is modelled by a cheap **ordered mask** in front of an MLP, and throughput
scales by batching many independent streams through one model. It is meant
as a base to grow toward the paper's numbers, not a claim to reproduce them.

## Architecture

```
sspc/
  coder.py   32-bit integer arithmetic coder (encoder/decoder, carry-free via pending bits)
  model.py   OrderedMaskMLP: embedding -> ordered mask -> 1 hidden ReLU layer -> softmax,
             hand-written backprop + Adam in float64 NumPy
  codec.py   container format + online loop: split input into N streams, per step
             predict all streams in one batch, code each byte, take one Adam step
  __main__.py CLI: compress / decompress / bench
tests/test_sspc.py  coder round trip, codec round trips (empty, 1 byte, random,
             text, more streams than bytes), learning sanity, CRC corruption
             detection, ordered-mask monotonicity, finite-difference gradient check
```

Ordered mask: `w_0 = 1`, `w_j = exp(-sum_{i<j} softplus(theta_i))`. Any value
of `theta` gives a strictly decreasing weight over distance, with only `k-1`
parameters.

Decisions:
- **NumPy, no PyTorch.** Encoder and decoder must produce bit-identical
  probabilities. Hand-written float64 math with fixed shapes is deterministic
  and easy to audit; the gradient is checked against finite differences.
- **Online training, no pretrained weights.** Keeps the format self-contained.
- **Probabilities quantised to 2^16 with a floor of 1 per symbol**, so a
  wrong prediction costs at most 16 bits and never breaks decoding.
- **CRC32 in the header** catches corruption and any float divergence between
  machines.

## Current results

`python -m sspc bench <51 KB of Python source>` (defaults: k=16, d=16,
hidden=256, lr=5e-3, 32 streams):

| codec   | bits/byte |
|---------|-----------|
| zlib-9  | 1.88 |
| bz2-9   | 1.81 |
| lzma-9e | 1.76 |
| sspc    | 2.56 (about 50 KiB/s enc and dec, round trip ok) |

On a 51 KB file the online model has only ~1.6k update steps, so it loses to
dictionary coders. Fewer streams compress better and run slower (16 streams:
2.48 bpb, ~32 KiB/s), which is the batch/throughput trade-off the paper
targets. Larger inputs give the model more steps; that has not been measured
here yet.

## Known limits

- Bit-exact decoding is only guaranteed on the same NumPy/BLAS build that
  encoded the file. A mismatch is detected by CRC, not silently wrong.
- Pure Python arithmetic coding loop; speed is far below a C/GPU coder.

## Next

1. Benchmark on enwik8 slices and standard corpora (Silesia) at 1-100 MB.
2. Port the model to PyTorch with deterministic kernels to get GPU batching,
   keeping the NumPy path as the reference for equivalence tests.
3. Ablate the ordered mask against no mask and a free per-position weight.
4. Move the coder loop to a vectorised or compiled implementation (rANS
   with one state per stream would fit the batched layout).
