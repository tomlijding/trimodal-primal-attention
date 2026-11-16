"""Trimodal primal attention block (Section 3 of the paper).

Notation follows the paper. Per head, for streams d = 1, 2, 3:

    Phi_d   = psi(X_d W^phi_d)                     N_d x m   features, rows L2-normalised
    W_d|X   = Phi_d^T softmax_N(tau_d Phi_d P_d)   m x R     data-dependent weights, eq. (4)
    B_d     = W_d|X G_d^T                          m x m     stream summary
    E_d     = Phi_d (B_f .* B_e) G_d               N_d x R   primal output, eq. (3)

with {d, e, f} = {1, 2, 3}. Neither the N^3 kernel tensor nor the core is formed, so the
cost is linear in every N_d, and the three streams may have different lengths.

Two switches select the block of the paper and its ablations (Table 1):

    pooling   "softmax"  eq. (4), parameter count independent of N          (ours)
              "linear"   W_d|X = Phi_d^T A_d, A_d in R^{N x R}              ("Lin. pooling")
              "static"   W_d a free parameter, independent of the input     ("Static W_d")
    core      "free"     G_d a free parameter, independent of the input     (ours)
              "gram"     G_d = W_d|X, the KMLSVD dual-solution core, so that
                         B_d = W_d W_d^T                                    ("W_d|X core")

`pooling="static"` only exists with `core="gram"`: the summaries are then constant and
E_d depends on stream d alone.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

POOLINGS = ("softmax", "linear", "static")
CORES = ("free", "gram")


class PrimalAttention(nn.Module):
    """The fusion operation: three streams in, three streams out, no residual or MLP.

    Args:
        dim: model width M.
        num_heads: H; the head width is m = M / H.
        rank: R, the number of columns of W_d|X and G_d.
        pooling, core: see the module docstring.
        seq_len: N, required only for pooling="linear" (its parameters grow with N).
        attn_drop, proj_drop: dropout before and after the output projection.
    """

    def __init__(self, dim, num_heads, rank, pooling="softmax", core="free", seq_len=None,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        if pooling not in POOLINGS or core not in CORES:
            raise ValueError(f"pooling must be in {POOLINGS} and core in {CORES}")
        if pooling == "static" and core != "gram":
            raise ValueError("pooling='static' requires core='gram'")
        if pooling == "linear" and seq_len is None:
            raise ValueError("pooling='linear' needs seq_len")
        self.dim, self.num_heads, self.head_dim, self.rank = dim, num_heads, dim // num_heads, rank
        self.pooling, self.core = pooling, core
        H, m, R = num_heads, self.head_dim, rank

        def per_stream(make):
            return nn.ParameterList(nn.Parameter(make()) for _ in range(3))

        def orthogonal():
            return nn.init.orthogonal_(torch.empty(H, m, R))

        # Output projection (after merging heads) and feature projection W^phi_d.
        self.proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(3))
        self.feature = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(3))

        if pooling == "static":
            self.W = per_stream(orthogonal)
        elif pooling == "linear":
            # Scale chosen so that the columns of W_d|X start near unit variance 1/R.
            std = math.sqrt(m / (seq_len * R))
            self.A = per_stream(lambda: torch.randn(H, seq_len, R) * std)
        else:
            self.P = per_stream(lambda: torch.randn(H, m, R) * m ** -0.5)
            # Learnable temperature tau_d, initialised at sqrt(m). Rows of Phi and columns
            # of P start near unit norm, so raw logits have sd ~ 1/sqrt(m); without the
            # scale the softmax over N tokens is indistinguishable from uniform.
            self.log_tau = per_stream(lambda: torch.tensor(math.log(math.sqrt(m))))
        if core == "free":
            # Unit-norm columns on average, matching the scale of W_d|X. Without this
            # gain the cross-stream path starts ~R times too weak and does not train.
            self.G = per_stream(lambda: orthogonal() * math.sqrt(R))

        # Learned projection of the R scores back to the head width m.
        self.projup = nn.ModuleList(nn.Linear(R, m) for _ in range(3))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def features(self, d, x):
        """Phi_d: B x N x H x m, rows L2-normalised per head."""
        B, N, _ = x.shape
        return F.normalize(self.feature[d](x).view(B, N, self.num_heads, self.head_dim), dim=-1)

    def weights(self, d, Phi, mask=None):
        """W_d|X: B x H x m x R with L2-normalised columns (H x m x R when static).

        mask: B x N bool, True = real token; padded tokens are excluded from the pooling.
        """
        if self.pooling == "static":
            return self.W[d]
        if self.pooling == "linear":
            if mask is not None:
                Phi = Phi * mask[:, :, None, None]
            W = torch.einsum("bnhm,hnr->bhmr", Phi, self.A[d])
        else:
            logits = torch.einsum("bnhm,hmr->bnhr", Phi, self.P[d]) * self.log_tau[d].exp()
            if mask is not None:
                logits = logits.masked_fill(~mask[:, :, None, None], float("-inf"))
            W = torch.einsum("bnhm,bnhr->bhmr", Phi, logits.softmax(dim=1))
        return F.normalize(W, dim=-2)

    def summary(self, d, W):
        """B_d, the m x m factor through which stream d reaches the other two."""
        if self.core == "free":
            return torch.einsum("bhmr,hkr->bhmk", W, self.G[d])
        return W @ W.transpose(-2, -1)

    def scores(self, x1, x2, x3, masks=None):
        """E_1, E_2, E_3 of eq. (3), each B x N_d x H x R."""
        masks = masks if masks is not None else (None, None, None)
        Phi = [self.features(d, x) for d, x in enumerate((x1, x2, x3))]
        W = [self.weights(d, Phi[d], masks[d]) for d in range(3)]
        Bs = [self.summary(d, W[d]) for d in range(3)]
        E = []
        for d, (e, f) in enumerate(((1, 2), (0, 2), (0, 1))):
            mix = Bs[f] * Bs[e]
            if self.pooling == "static":
                E.append(torch.einsum("bnhm,hmk,hkr->bnhr", Phi[d], mix, W[d]))
            else:
                # Trailing factor: the free core G_d, or W_d|X itself for the Gram core.
                G = self.G[d].expand(mix.shape[0], -1, -1, -1) if self.core == "free" else W[d]
                E.append(torch.einsum("bnhm,bhmk,bhkr->bnhr", Phi[d], mix, G))
        return E

    def forward(self, x1, x2, x3, masks=None):
        out = []
        for d, E in enumerate(self.scores(x1, x2, x3, masks)):
            y = self.projup[d](E).flatten(2)                    # B x N_d x M
            out.append(self.proj_drop(self.proj[d](self.attn_drop(y))))
        return out


class PrimalBlock(nn.Module):
    """Pre-norm transformer block per stream, with PrimalAttention in place of self-attention."""

    def __init__(self, dim, num_heads, rank, pooling="softmax", core="free", seq_len=None,
                 mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        self.attn = PrimalAttention(dim, num_heads, rank, pooling=pooling, core=core,
                                    seq_len=seq_len, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.ModuleList(nn.LayerNorm(dim) for _ in range(3))
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(proj_drop),
                          nn.Linear(hidden, dim), nn.Dropout(proj_drop))
            for _ in range(3))

    def forward(self, x1, x2, x3, masks=None):
        xs = (x1, x2, x3)
        out = self.attn(*[n(x) for n, x in zip(self.norm1, xs)], masks=masks)
        xs = [x + o for x, o in zip(xs, out)]
        return [x + mlp(n(x)) for x, mlp, n in zip(xs, self.mlp, self.norm2)]
