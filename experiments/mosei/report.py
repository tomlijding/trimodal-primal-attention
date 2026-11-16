"""Table 2 from MMSA stdout logs: test metrics per arm, mean ± SD over seeds, and paired
t-tests against MulT.

    python experiments/mosei/report.py results/mosei/logs

Every `<arm>*.log` in the directory is read; a log may hold one seed or several. For each
seed the test result is the one MMSA prints ("Result for seed"), i.e. the test metrics of
the checkpoint with the lowest validation loss.
"""
import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARMS = {"lmf": "LMF", "mult": "MulT", "mult_nofusion": "MulT w/o fusion", "primal": "Primal block (ours)"}
# column header, key in MMSA's result dict, scale
COLS = [("MAE", "MAE", 1), ("Corr", "Corr", 1), ("Acc2 neg/non", "Has0_acc_2", 100),
        ("Acc2 neg/pos", "Non0_acc_2", 100), ("F1 neg/non", "Has0_F1_score", 100),
        ("F1 neg/pos", "Non0_F1_score", 100), ("Acc7", "Mult_acc_7", 100)]
RES = re.compile(r"Result for seed (\d+): (\{.*\})")


def load(log_dir):
    arms = defaultdict(dict)
    for f in sorted(Path(log_dir).glob("*.log")):
        arm = re.sub(r"_s\d+$", "", f.stem)
        for m in RES.finditer(f.read_text(errors="ignore")):
            arms[arm][int(m.group(1))] = ast.literal_eval(m.group(2))
    return arms


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("log_dir")
    p.add_argument("--ref", default="mult")
    a = p.parse_args()
    arms = load(a.log_dir)
    order = [k for k in ARMS if k in arms] + [k for k in arms if k not in ARMS]

    print(f"{'arm':<22}{'n':>3}" + "".join(f"{c[0]:>18}" for c in COLS))
    for k in order:
        r = arms[k]
        cells = ""
        for _, key, scale in COLS:
            x = np.array([v[key] for v in r.values()]) * scale
            fmt = ".3f" if scale == 1 else ".1f"
            cells += f"{f'{x.mean():{fmt}} ± {x.std(ddof=1):{fmt}}':>18}"
        print(f"{ARMS.get(k, k):<22}{len(r):>3}{cells}")
    print("(mean ± sample SD over seeds)")

    ref = arms.get(a.ref, {})
    print(f"\npaired t-test against {ARMS.get(a.ref, a.ref)} on common seeds")
    for k in order:
        common = sorted(set(arms[k]) & set(ref))
        if k == a.ref or len(common) < 2:
            continue
        parts = []
        for name, key, scale in (COLS[0], COLS[3]):
            x = np.array([arms[k][s][key] for s in common]) * scale
            y = np.array([ref[s][key] for s in common]) * scale
            parts.append(f"{name}: diff {np.mean(x - y):+.4f}, p = {stats.ttest_rel(x, y).pvalue:.2f}")
        print(f"  {ARMS.get(k, k):<22}" + "   ".join(parts))


if __name__ == "__main__":
    main()
