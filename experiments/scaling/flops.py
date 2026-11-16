"""FLOPs of one fusion operation: primal block vs MulT's pairwise cross-attention, and the
memory limit of the explicit score tensor (Section 3, "Computational and memory cost").

Configuration of Section 4.2: the primal operation at M = 60, H = 6, R = m = 10, and MulT's
cross-modal attention at its own width d = 30 (six directed pairs). All three streams have
length N. Counted per sample, in multiply-accumulates (MACs); elementwise operations,
softmax, normalisation and biases are not counted, for either method. FLOPs = 2 x MACs,
the convention of torch.utils.flop_counter.

    primal   6 N M^2 + 12 N M R + 3 N M m + 3 H m^2 R
               features and output projection   6 N M^2
               pooling (logits, weighted sum)   6 N M R
               summaries B_d = W_d G_d^T        3 H m^2 R
               contraction Phi_d (B_f .* B_e) G_d   3 N M m + 3 N M R
               projection of the R scores to m  3 N M R
    MulT     6 x (4 N d^2 + 2 N^2 d)
               q, k, v and output projections   4 N d^2   per directed pair
               scores Q K^T and weighted sum    2 N^2 d   per directed pair

Both closed forms are checked for equality against torch.utils.flop_counter, on the
block itself and on a reference implementation of the six cross-attentions.

    python experiments/scaling/flops.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from primal_block import PrimalAttention  # noqa: E402

# Contract einsums left to right, as written in the block (apply Phi_d to the summaries
# first). With opt_einsum installed PyTorch may fold G_d into the summaries first, which
# replaces the 3 N M m term by a second 3 H m^2 R and is cheaper still for N > R.
import torch.backends.opt_einsum  # noqa: E402

torch.backends.opt_einsum.enabled = False

M, H, R = 60, 6, 10      # primal operation
D_MULT, H_MULT = 30, 6   # MulT cross-modal attention
BUDGET, BATCH = 24 * 2**30, 32


def primal_macs(N, M=M, H=H, R=R):
    m = M // H
    return 6 * N * M * M + 12 * N * M * R + 3 * N * M * m + 3 * H * m * m * R


def mult_macs(N, d=D_MULT):
    return 6 * (4 * N * d * d + 2 * N * N * d)


class PairwiseCrossAttention(nn.Module):
    """Six directed cross-attentions with separate q, k, v and output projections, as in
    MulT's cross-modal layer. Used only to check mult_macs."""
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.ModuleList(nn.Linear(d, d) for _ in range(6))
                                          for _ in range(4))

    def forward(self, x1, x2, x3):
        xs, out, i = (x1, x2, x3), [], 0
        for t in range(3):
            for s in range(3):
                if s == t:
                    continue
                B, N, d = xs[t].shape
                split = lambda z: z.view(B, -1, self.h, d // self.h).transpose(1, 2)
                q, k, v = split(self.q[i](xs[t])), split(self.k[i](xs[s])), split(self.v[i](xs[s]))
                a = (q @ k.transpose(-2, -1)).softmax(-1)
                out.append(self.o[i]((a @ v).transpose(1, 2).reshape(B, N, d)))
                i += 1
        return out


def measured(module, N, width):
    xs = [torch.randn(1, N, width) for _ in range(3)]
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        module(*xs)
    return fc.get_total_flops()


def main():
    torch.manual_seed(0)
    primal = PrimalAttention(M, H, R).eval()
    mult = PairwiseCrossAttention(D_MULT, H_MULT).eval()

    print("FLOPs per sample, closed form vs torch.utils.flop_counter")
    print(f"{'N':>6}{'primal closed':>16}{'measured':>16}{'MulT closed':>16}{'measured':>16}")
    for N in (16, 100, 500):
        pc, mc = 2 * primal_macs(N), 2 * mult_macs(N)
        pm, mm = measured(primal, N, M), measured(mult, N, D_MULT)
        print(f"{N:>6}{pc:>16,}{pm:>16,}{mc:>16,}{mm:>16,}")
        assert pm == pc and mm == mc, "closed form does not match the FLOP counter"

    print("\nFLOPs per sample, MulT pairwise cross-attention vs primal operation")
    print(f"{'N':>6}{'primal':>16}{'MulT':>18}{'ratio':>8}")
    for N in (50, 500, 1000, 4000, 16000):
        p, m = 2 * primal_macs(N), 2 * mult_macs(N)
        print(f"{N:>6}{p:>16,}{m:>18,}{m / p:>7.1f}x")

    # Explicit score tensor: one float32 N x N x N tensor per sample.
    n = int((BUDGET / (4 * BATCH)) ** (1 / 3))
    while BATCH * 4 * (n + 1) ** 3 <= BUDGET:
        n += 1
    print(f"\nExplicit N^3 score tensor, float32, batch {BATCH}: largest N within 24 GiB = {n}")


if __name__ == "__main__":
    main()
