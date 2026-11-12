"""Ordered-mask MLP byte predictor with hand-written backprop (NumPy, float64).

Model (one row per parallel stream):

    ctx[j]  = byte at distance j+1 in the past, j = 0..k-1
    w_j     = exp(-sum_{i<j} softplus(theta_i))     # ordered mask, w_0 = 1
    x       = concat_j( w_j * E[ctx[j]] )           # (k*d,)
    h       = relu(x @ W1 + b1)
    logits  = h @ W2 + b2                           # (256,)

The ordered mask is the "simple parameterization" of history importance: the
cumulative-softplus form guarantees w_0 >= w_1 >= ... >= w_{k-1} > 0 for any
theta, so nearer bytes can never be weighted below farther ones, while k - 1
scalars learn how fast importance decays.

All arithmetic is deterministic for fixed input shapes, which is what lets the
decoder replay the encoder's online training bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VOCAB = 256


@dataclass(frozen=True)
class ModelConfig:
    history: int = 16  # k
    embed: int = 16  # d
    hidden: int = 256
    lr: float = 2e-3
    seed: int = 0


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class OrderedMaskMLP:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        k, d, h = cfg.history, cfg.embed, cfg.hidden
        rng = np.random.default_rng(cfg.seed)
        self.params: dict[str, np.ndarray] = {
            "E": rng.normal(0.0, 1.0, (VOCAB, d)),
            "theta": np.full(max(k - 1, 0), -3.0),
            "W1": rng.normal(0.0, np.sqrt(2.0 / (k * d)), (k * d, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(0.0, np.sqrt(1.0 / h), (h, VOCAB)),
            "b2": np.zeros(VOCAB),
        }
        self._m = {n: np.zeros_like(p) for n, p in self.params.items()}
        self._v = {n: np.zeros_like(p) for n, p in self.params.items()}
        self._t = 0

    # ---- forward -------------------------------------------------------
    def mask(self) -> np.ndarray:
        theta = self.params["theta"]
        cum = np.concatenate(([0.0], np.cumsum(_softplus(theta))))
        return np.exp(-cum)

    def _forward(self, ctx: np.ndarray):
        p = self.params
        b = ctx.shape[0]
        w = self.mask()
        emb = p["E"][ctx]  # (B, k, d)
        x = (emb * w[None, :, None]).reshape(b, -1)
        pre = x @ p["W1"] + p["b1"]
        hid = np.maximum(pre, 0.0)
        logits = hid @ p["W2"] + p["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob, (ctx, w, emb, x, pre, hid)

    def predict(self, ctx: np.ndarray) -> np.ndarray:
        """Next-byte distributions, shape (B, 256)."""
        return self._forward(ctx)[0]

    # ---- backward ------------------------------------------------------
    def gradients(self, ctx, target, active, cache=None, prob=None):
        """Mean cross-entropy over active rows and its gradients."""
        if cache is None:
            prob, cache = self._forward(ctx)
        ctx, w, emb, x, pre, hid = cache
        p = self.params
        b, k = ctx.shape
        n = max(int(active.sum()), 1)
        rows = np.arange(b)
        loss = -np.log(prob[rows, target][active > 0]).sum() / n

        dlogits = prob.copy()
        dlogits[rows, target] -= 1.0
        dlogits *= (active / n)[:, None]
        g = {
            "W2": hid.T @ dlogits,
            "b2": dlogits.sum(axis=0),
        }
        dpre = (dlogits @ p["W2"].T) * (pre > 0)
        g["W1"] = x.T @ dpre
        g["b1"] = dpre.sum(axis=0)
        dx = (dpre @ p["W1"].T).reshape(b, k, -1)
        dE = np.zeros_like(p["E"])
        np.add.at(dE, ctx, dx * w[None, :, None])
        g["E"] = dE
        dw = (dx * emb).sum(axis=(0, 2))  # (k,)
        dcum = -w * dw  # w = exp(-cum)
        # cum_j = sum_{i<j} softplus(theta_i)  ->  dsp_i = sum_{j>i} dcum_j
        dsp = np.cumsum(dcum[::-1])[::-1][1:]
        g["theta"] = dsp * _sigmoid(p["theta"])
        return loss, g

    def train_step(self, ctx, target, active, cache=None, prob=None) -> float:
        loss, g = self.gradients(ctx, target, active, cache, prob)
        self._t += 1
        lr, b1, b2, eps = self.cfg.lr, 0.9, 0.999, 1e-8
        c1 = 1.0 - b1**self._t
        c2 = 1.0 - b2**self._t
        for name, grad in g.items():
            m, v = self._m[name], self._v[name]
            m *= b1
            m += (1 - b1) * grad
            v *= b2
            v += (1 - b2) * grad * grad
            self.params[name] -= lr * (m / c1) / (np.sqrt(v / c2) + eps)
        return loss

    def step(self, ctx, target, active) -> float:
        """Forward + backward + Adam update, reusing one forward pass."""
        prob, cache = self._forward(ctx)
        return self.train_step(ctx, target, active, cache, prob)
