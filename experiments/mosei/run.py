"""Train one arm of Table 2 on CMU-MOSEI with MMSA (see README.md in this directory for setup).

    python run.py primal --data-root /path/to/MMSA --gpu 0 > logs/primal.log 2>&1
    python report.py logs

Arms
    mult           MulT as implemented in MMSA (six pairwise cross-modal transformers)
    mult_nofusion  the same six transformers as self-attention: no stream sees another
    primal         four trimodal primal blocks in place of the cross-modal stage
    lmf            MMSA's LMF, with its own front end and published configuration

--data-root is the directory that contains MOSEI/Processed/unaligned_50.pkl.

The three MulT-skeleton arms use MulT's published MOSEI configuration, except that the
batch of 32 is processed in one step: MMSA's default is batch 4 with gradients accumulated
over 8 steps (update_epochs = 8) and a clipping threshold of 0.6 on the summed gradient;
here batch_size = 32, update_epochs = 1 and grad_clip = 0.6 / 8.
"""
import argparse
import os
from pathlib import Path

MULT_COMMON = {"batch_size": 32, "update_epochs": 1, "grad_clip": 0.075}
ARMS = {
    "mult": ("mult", {**MULT_COMMON}),
    "mult_nofusion": ("mult", {**MULT_COMMON, "fusion": "nocross"}),
    "primal": ("mult", {**MULT_COMMON, "fusion": "primal"}),
    "lmf": ("lmf", {}),
}

p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
p.add_argument("arm", choices=sorted(ARMS))
p.add_argument("--data-root", required=True, type=Path)
p.add_argument("--seeds", type=int, nargs="+", default=[1111, 1112, 1113, 1114, 1115])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--out", type=Path, default=Path("runs"), help="checkpoints and MMSA's own logs")
a = p.parse_args()

# Expose only the requested GPU, before PyTorch initialises CUDA. MMSA queries every visible
# device at start-up, so one faulty card would otherwise stop the run. --gpu is an index in
# PCI bus order, which is normally nvidia-smi's numbering. If CUDA_VISIBLE_DEVICES is
# already set, it is left alone and --gpu indexes into it.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    a.gpu = 0

from MMSA import MMSA_run, get_config_regression  # noqa: E402

model, overrides = ARMS[a.arm]
config = get_config_regression(model, "mosei")
config["featurePath"] = str(a.data_root / "MOSEI" / "Processed" / "unaligned_50.pkl")
config.update(overrides)
for k, v in overrides.items():
    print(f"[run] set {k} = {v!r}", flush=True)

tag = f"{a.arm}_" + "_".join(map(str, a.seeds))
MMSA_run(model, "mosei", config=config, seeds=a.seeds, gpu_ids=[a.gpu], num_workers=2,
         model_save_dir=a.out / "saved_models" / tag, res_save_dir=a.out / "results" / tag,
         log_dir=a.out / "logs")
