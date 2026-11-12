# sspc: online learned lossless compression with an ordered-mask MLP

Companion code for *Accelerating General-Purpose Lossless Compression via Simple
Scalable Parameterization* (Yu Mao, Yufei Cui, Tei-Wei Kuo, Chun Jason Xue,
ACM Multimedia 2022). [Paper PDF](https://mynotwo.github.io/assets/pdfs/accelerating-lossless.pdf)

This is a compact, dependency-light reimplementation of the idea, not the
original experiment code: a byte-level MLP predictor whose input is the last
`k` bytes weighted by a learned **ordered mask** (importance that can only
decay with distance), driving an arithmetic coder. The model starts from a
fixed seed and trains online while coding, so the compressed file needs no
stored weights; the decoder replays the same training.

## Quick start

```bash
pip install -e '.[test]'      # only needs numpy
python -m pytest -q           # 11 tests
python -m sspc compress  input.bin out.sspc
python -m sspc decompress out.sspc restored.bin
python -m sspc bench input.bin            # vs zlib / bz2 / lzma, checks round trip
```

Model knobs: `--history --embed --hidden --lr --seed --streams`. They are
recorded in the file header, so `decompress` needs no flags.

See [REDESIGN.md](REDESIGN.md) for the design and current numbers.
