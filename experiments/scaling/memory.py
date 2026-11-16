"""Measured peak GPU memory of the fusion stage: MulT's six cross-modal transformers vs the
four primal blocks that replace them (Section 3, "Computational and memory cost").

Needs the patched MMSA of experiments/mosei (no data: the inputs are random sequences).
Both stages are built from MulT's published MOSEI configuration, fed three streams of equal
length N at batch 32, and run forward and backward. Reported: peak allocated memory at the
given lengths, and the largest N that fits on the device (bisection on out-of-memory).

    python experiments/scaling/memory.py --out results/scaling/memory.json
"""
import argparse
import json
import os
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lengths", type=int, nargs="+", default=[50, 100, 200, 500])
p.add_argument("--n-max", type=int, default=64000, help="upper end of the out-of-memory search")
p.add_argument("--out", default=None)
a = p.parse_args()

# Expose only the requested GPU before PyTorch initialises CUDA, as in experiments/mosei/run.py.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    a.gpu = 0

import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402

from MMSA.config import get_config_regression  # noqa: E402
from MMSA.models.singleTask.MULT import MULT  # noqa: E402

dev = torch.device(f"cuda:{a.gpu}")


def build(fusion):
    cfg = get_config_regression("mult", "mosei")
    model = MULT(EasyDict(dict(cfg, fusion=fusion, train_mode="regression"))).to(dev).train()
    width = model.proj_l.out_channels
    if fusion == "primal":
        return model, width, model._primal_fuse

    def crossmodal(l, au, v):
        return (model.trans_l_with_a(l, au, au), model.trans_l_with_v(l, v, v),
                model.trans_a_with_l(au, l, l), model.trans_a_with_v(au, v, v),
                model.trans_v_with_l(v, l, l), model.trans_v_with_a(v, au, au))
    return model, width, crossmodal


def peak_gib(model, width, stage, N):
    """Peak allocated GiB of one forward and backward pass, or None on out-of-memory."""
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        xs = [torch.randn(N, a.batch, width, device=dev, requires_grad=True) for _ in range(3)]
        sum(o.float().sum() for o in stage(*xs)).backward()
        return torch.cuda.max_memory_allocated(dev) / 2**30
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        xs = None


def largest_n(model, width, stage, lo, hi):
    """Largest N in [lo, hi] that does not run out of memory, to within 1%."""
    if peak_gib(model, width, stage, hi) is not None:
        return hi
    while hi - lo > max(1, lo // 100):
        mid = (lo + hi) // 2
        if peak_gib(model, width, stage, mid) is None:
            hi = mid
        else:
            lo = mid
    return lo


total = torch.cuda.get_device_properties(dev).total_memory / 2**30
res = {"device": torch.cuda.get_device_name(dev), "device_GiB": round(total, 2), "batch": a.batch}
print(f"{res['device']}, {total:.1f} GiB, batch {a.batch}, forward + backward")
for fusion, name in (("crossmodal", "MulT cross-modal stage"), ("primal", "primal blocks")):
    model, width, stage = build(fusion)
    peaks = {N: peak_gib(model, width, stage, N) for N in a.lengths}
    ok = [N for N, g in peaks.items() if g is not None]
    limit = largest_n(model, width, stage, max(ok), a.n_max)
    res[fusion] = {"width": width, "peak_GiB": {N: g and round(g, 3) for N, g in peaks.items()},
                   "largest_N": limit}
    cells = "  ".join(f"N={N}: " + ("OOM" if g is None else f"{g:.2f} GiB") for N, g in peaks.items())
    print(f"{name:<24} {cells}  largest N that fits: {limit}", flush=True)
    del model, stage
    torch.cuda.empty_cache()
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
