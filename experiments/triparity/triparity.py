"""TriParity (Section 4.1, Table 1): synthetic targets of controlled interaction order.

Three streams of N = 16 tokens; every token of stream d is a noisy copy of one latent u_d.
With p_d = a_d^T u_d, the order-k label is the sign of the degree-k elementary symmetric
polynomial of (p_1, p_2, p_3):

    order 1   p1 + p2 + p3              additive
    order 2   p1 p2 + p2 p3 + p1 p3     pairwise
    order 3   p1 p2 p3                  three-way parity

Every mechanism is one fusion layer at width 64, followed by mean pooling and a linear
head, trained for 3000 Adam steps on 20k samples and scored on 5k held-out samples.

    python experiments/triparity/triparity.py --out results/triparity/table1.json
    python experiments/triparity/triparity.py --rows ours,static --orders 3 --seeds 1
    python experiments/triparity/triparity.py --show results/triparity/table1.json

A full run (10 rows x 3 orders x 3 seeds) takes about 45 minutes on one GPU.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from primal_block import PrimalBlock  # noqa: E402

D, N, HEADS, RANK, LATENT = 64, 16, 4, 16, 8

# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

# The task (embedding matrices M_d and target directions a_d) is drawn once and shared by
# the training and held-out sets.
_TASK = torch.Generator().manual_seed(1234)
M_TASK = torch.randn(3, LATENT, D, generator=_TASK) / math.sqrt(LATENT)
A_TASK = torch.randn(3, LATENT, generator=_TASK)


def make_data(n, order, device, noise=0.5, seed=0):
    """Returns ([x1, x2, x3], y) with x_d of shape n x N x D and y in {0, 1}."""
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(3, n, LATENT, generator=g)
    x = [(u[d] @ M_TASK[d]).unsqueeze(1).expand(n, N, D)
         + noise * torch.randn(n, N, D, generator=g) for d in range(3)]
    p = torch.stack([u[d] @ A_TASK[d] for d in range(3)])
    if order == 1:
        s = p.sum(0)
    elif order == 2:
        s = p[0] * p[1] + p[1] * p[2] + p[0] * p[2]
    else:
        s = p[0] * p[1] * p[2]
    return [t.to(device) for t in x], (s > 0).float().to(device)


def pool(t):
    return t.mean(1)


def mlp():
    return nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))


# ---------------------------------------------------------------------------------------
# Mechanisms. Each maps (x1, x2, x3) to one logit per sample.
# ---------------------------------------------------------------------------------------

class LateFusion(nn.Module):
    """No fusion: a per-stream MLP on the pooled tokens, summed. Solves order 1 only."""
    def __init__(self, h=96):
        super().__init__()
        self.f = nn.ModuleList(nn.Sequential(nn.Linear(D, h), nn.GELU(), nn.Linear(h, h))
                               for _ in range(3))
        self.o = nn.Linear(h, 1)

    def forward(self, x1, x2, x3):
        return self.o(sum(f(pool(x)) for f, x in zip(self.f, (x1, x2, x3)))).squeeze(-1)


class LMF(nn.Module):
    """Low-rank multimodal fusion (Liu et al., ACL 2018) on mean-pooled tokens:
    z = sum_r prod_d (W_d^r [h_d; 1]), then a linear head. Rank 10, width 64."""
    def __init__(self, rank=10, h=64):
        super().__init__()
        self.f = nn.Parameter(torch.randn(3, rank, D + 1, h) * (D + 1) ** -0.5)
        self.w = nn.Parameter(torch.full((rank,), 1.0 / rank))
        self.o = nn.Linear(h, 1)

    def forward(self, x1, x2, x3):
        z = 1
        for d, x in enumerate((x1, x2, x3)):
            h = torch.cat([pool(x), torch.ones_like(pool(x)[:, :1])], -1)
            z = z * torch.einsum("bi,rih->brh", h, self.f[d])
        return self.o(torch.einsum("brh,r->bh", z, self.w)).squeeze(-1)


class TensorFormer(nn.Module):
    """TensorFormer attention (Sun, Chen & Lin, IEEE TAFFC 2023), one layer, in the same
    pre-norm block and head as the primal rows.

    Each stream is projected to width D and averaged over the feature dimension, giving one
    scalar per token; the attention tensor is the outer product A_ijk = a_i b_j c_k, which
    is materialised (N^3). For stream d, a softmax over stream d's axis weights its
    projected tokens; the result is averaged over the other two axes into one vector, which
    is added to every token of stream d.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(D, D) for _ in range(3))
        self.val = nn.ModuleList(nn.Linear(D, D) for _ in range(3))
        self.out = nn.ModuleList(nn.Linear(D, D) for _ in range(3))
        self.n1 = nn.ModuleList(nn.LayerNorm(D) for _ in range(3))
        self.n2 = nn.ModuleList(nn.LayerNorm(D) for _ in range(3))
        self.m = nn.ModuleList(mlp() for _ in range(3))
        self.o = nn.Linear(3 * D, 1)

    def attention(self, xs):
        bar = [p(x).mean(-1) for p, x in zip(self.proj, xs)]                  # 3 x (B, N)
        A = torch.einsum("bi,bj,bk->bijk", *bar)                             # B x N x N x N
        outs = []
        for d in range(3):
            w = A.softmax(dim=1 + d).mean(dim=[a for a in (1, 2, 3) if a != 1 + d])
            v = torch.einsum("bn,bnm->bm", w, self.val[d](xs[d]))
            outs.append(self.out[d](v).unsqueeze(1).expand_as(xs[d]))
        return outs

    def forward(self, x1, x2, x3):
        xs = [x1, x2, x3]
        xs = [x + o for x, o in zip(xs, self.attention([n(x) for n, x in zip(self.n1, xs)]))]
        xs = [x + m(n(x)) for x, m, n in zip(xs, self.m, self.n2)]
        return self.o(torch.cat([pool(x) for x in xs], -1)).squeeze(-1)


class MMT(nn.Module):
    """MMT multi-way attention (Tang et al., IJCAI 2022, eqs. 1-6), one layer, as printed:
    no softmax, no normalisation, no value projection.

    Per modality i, the query-key core is G[t, w, s] = (x_t Q1 * x_t K1)_w (x_t Q2 * x_t K2)_s
    and M_i = mean_t G. The ring is closed by contracting modality i's core with the pooled
    matrices of the other two; then atten_i = linear(.) and Y_i = atten_i * X_i + a X_i
    (a is a fixed constant in the paper, learned here).
    """
    def __init__(self, r=RANK):
        super().__init__()
        self.w = nn.ModuleList(nn.Linear(D, 4 * r, bias=False) for _ in range(3))
        self.lin = nn.ModuleList(nn.Linear(r * r, D) for _ in range(3))
        self.a = nn.Parameter(torch.tensor(0.5))
        self.o = nn.Linear(3 * D, 1)

    def core(self, i, x):
        q1, q2, k1, k2 = self.w[i](x).chunk(4, -1)
        return torch.einsum("btw,bts->btws", q1 * k1, q2 * k2)

    def forward(self, *xs):
        g = [self.core(i, x) for i, x in enumerate(xs)]
        m = [c.mean(1) for c in g]
        att = [self.lin[i](torch.einsum("btws,bsu,buv->btwv",
                                        g[i], m[(i + 1) % 3], m[(i + 2) % 3]).flatten(2))
               for i in range(3)]
        ys = [(at + self.a) * x for at, x in zip(att, xs)]
        return self.o(torch.cat([pool(y) for y in ys], -1)).squeeze(-1)


class PairwiseCrossAttention(nn.Module):
    """The cross-modal layer of MulT (Tsai et al., ACL 2019): six directed pairwise
    cross-attentions, one per ordered pair of streams. Each stream adds the outputs of the
    two attentions in which it is the query to its residual. No feed-forward sublayer."""
    def __init__(self):
        super().__init__()
        self.at = nn.ModuleList(nn.MultiheadAttention(D, HEADS, batch_first=True)
                                for _ in range(6))
        self.n = nn.LayerNorm(D)
        self.o = nn.Linear(3 * D, 1)

    def forward(self, x1, x2, x3):
        xs, outs, k = [x1, x2, x3], [], 0
        for i in range(3):
            acc = xs[i]
            for j in range(3):
                if i == j:
                    continue
                q, kv = self.n(xs[i]), self.n(xs[j])
                acc = acc + self.at[k](q, kv, kv, need_weights=False)[0]
                k += 1
            outs.append(acc)
        return self.o(torch.cat([pool(z) for z in outs], -1)).squeeze(-1)


class ConcatSelfAttention(nn.Module):
    """One self-attention block over the 3N concatenated tokens."""
    def __init__(self):
        super().__init__()
        self.at = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.n = nn.LayerNorm(D)
        self.m = mlp()
        self.o = nn.Linear(D, 1)

    def forward(self, x1, x2, x3):
        z = torch.cat([x1, x2, x3], 1)
        zn = self.n(z)
        z = z + self.at(zn, zn, zn, need_weights=False)[0]
        z = z + self.m(z)
        return self.o(pool(z)).squeeze(-1)


class Primal(nn.Module):
    """One PrimalBlock (H = 4, R = m = 16), mean pooling per stream, linear head."""
    def __init__(self, pooling, core):
        super().__init__()
        self.b = PrimalBlock(D, HEADS, RANK, pooling=pooling, core=core, seq_len=N)
        self.o = nn.Linear(3 * D, 1)

    def forward(self, x1, x2, x3):
        return self.o(torch.cat([pool(x) for x in self.b(x1, x2, x3)], -1)).squeeze(-1)


# key: (label as in Table 1, constructor)
ROWS = {
    "late":         ("Late fusion", LateFusion),
    "lmf":          ("LMF on pooled tokens", LMF),
    "tensorformer": ("TensorFormer", TensorFormer),
    "mmt":          ("MMT", MMT),
    "mult":         ("MulT", PairwiseCrossAttention),
    "concat":       ("Concat self-att.", ConcatSelfAttention),
    "ours":         ("Primal block (ours)", lambda: Primal("softmax", "free")),
    "linear":       ("Lin. pooling", lambda: Primal("linear", "free")),
    "gram":         ("W_d|X core", lambda: Primal("softmax", "gram")),
    "static":       ("Static W_d", lambda: Primal("static", "gram")),
}


# ---------------------------------------------------------------------------------------
# Training and reporting
# ---------------------------------------------------------------------------------------

def train_eval(make, x, y, xv, yv, steps=3000, bs=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = make().to(y.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = y.shape[0]
    for _ in range(steps):
        i = torch.randint(0, n, (bs,), device=y.device)
        loss = F.binary_cross_entropy_with_logits(model(*[z[i] for z in x]), y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return ((model(*xv) > 0).float() == yv).float().mean().item()


def show(res):
    orders = sorted({o for r in res.values() for o in r["acc"]})
    names = {"1": "Additive", "2": "Pairwise", "3": "Three-way"}
    print(f"\n{'mechanism':<22}{'params':>8}" + "".join(f"{names[o]:>16}" for o in orders))
    for key, r in res.items():
        cells = "".join(f"{np.mean(r['acc'][o]):>9.3f} ±{np.std(r['acc'][o]):.3f}"
                        if o in r["acc"] else f"{'':>16}" for o in orders)
        print(f"{ROWS[key][0]:<22}{r['params'] / 1e3:>7.0f}k{cells}")
    print("(mean ± SD over seeds)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", default="all", help="comma-separated subset of: " + ", ".join(ROWS))
    ap.add_argument("--orders", default="1,2,3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0, help="index of the GPU to use")
    ap.add_argument("--device", default=None, help="e.g. cpu; default: the GPU given by --gpu, else cpu")
    ap.add_argument("--out", default=None, help="write per-seed accuracies to this JSON file")
    ap.add_argument("--show", default=None, help="print the table of an existing JSON file and exit")
    a = ap.parse_args()

    if a.show:
        show(json.load(open(a.show)))
        return

    if a.device is None:
        # Expose only the requested GPU before CUDA is initialised, so that a faulty card
        # elsewhere in the machine cannot stop the run.
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
            a.gpu = 0
        a.device = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"device: {a.device}", flush=True)

    rows = list(ROWS) if a.rows == "all" else a.rows.split(",")
    res = {k: {"params": sum(p.numel() for p in ROWS[k][1]().parameters()), "acc": {}}
           for k in rows}
    t0 = time.time()
    for order in (int(o) for o in a.orders.split(",")):
        x, y = make_data(20000, order, a.device, seed=10 + order)
        xv, yv = make_data(5000, order, a.device, seed=90 + order)
        for k in rows:
            accs = [train_eval(ROWS[k][1], x, y, xv, yv, seed=s) for s in range(a.seeds)]
            res[k]["acc"][str(order)] = accs
            print(f"order {order}  {ROWS[k][0]:<22} {np.mean(accs):.3f} ± {np.std(accs):.3f}"
                  f"   [{time.time() - t0:.0f}s]", flush=True)
            if a.out:
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                json.dump(res, open(a.out, "w"), indent=1)
    show(res)


if __name__ == "__main__":
    main()
