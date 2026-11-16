"""Numerical check of the two structural claims of Section 3, at initialisation, on the
fusion operation alone (no residual, no MLP). Runs on CPU in a few seconds.

  1. Coupling requires data-dependent weights: with static W_d, the output of stream 1
     does not change when stream 2 is replaced.
  2. The KMLSVD (Gram) core makes the output an even function of every other stream:
     exactly with linear pooling, approximately with softmax pooling. A free core makes
     it odd (exactly with linear pooling), which a triple product p1 p2 p3 requires.

For f = stream 1's output:
    coupling  = std[f(x1, x2, x3) - f(x1, x2', x3)]             x2' an independent resample
    odd share = std[(f(x1, x2, x3) - f(x1, -x2, x3)) / 2] / coupling * sqrt(2)
so that 0 = sign-blind (even) and 1 = fully sign-sensitive (odd) in stream 2.

    python experiments/triparity/parity_check.py
"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from primal_block import PrimalAttention  # noqa: E402

D, N, HEADS, RANK, B = 64, 16, 4, 16, 2048
VARIANTS = [
    ("Primal block (ours): softmax pooling, free core", "softmax", "free"),
    ("Lin. pooling: linear pooling, free core", "linear", "free"),
    ("W_d|X core: softmax pooling, Gram core", "softmax", "gram"),
    ("linear pooling, Gram core", "linear", "gram"),
    ("Static W_d", "static", "gram"),
]


def main():
    g = torch.Generator().manual_seed(0)
    x1, x2, x3, x2p = (torch.randn(B, N, D, generator=g, dtype=torch.float64) for _ in range(4))
    print(f"{'variant':<50}{'coupling':>10}{'odd share':>11}")
    for name, pooling, core in VARIANTS:
        torch.manual_seed(0)
        op = PrimalAttention(D, HEADS, RANK, pooling=pooling, core=core, seq_len=N).double().eval()
        with torch.no_grad():
            f = lambda b: op(x1, b, x3)[0]
            base = f(x2)
            coupling = (base - f(x2p)).std().item()
            odd = ((base - f(-x2)) / 2).std().item()
        share = "   n/a" if coupling < 1e-12 else f"{odd / coupling * math.sqrt(2):.3f}"
        print(f"{name:<50}{coupling:>10.2e}{share:>11}")


if __name__ == "__main__":
    main()
