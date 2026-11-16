# CMU-MOSEI (Table 2)

The MOSEI experiment runs inside [MMSA](https://github.com/thuiar/MMSA) (MIT licence). MMSA is
not vendored here: `mmsa.patch` is the complete change to it, one file
(`src/MMSA/models/singleTask/MULT.py`). The patch adds a `fusion` option to MMSA's MulT:

| `fusion` | Cross-modal stage | Table 2 row |
|---|---|---|
| `crossmodal` (default) | MulT's six pairwise cross-modal transformers, unchanged | MulT |
| `nocross` | the same six transformers as self-attention, so no stream sees another | MulT w/o fusion |
| `primal` | four `PrimalBlock`s over the three streams jointly (M = 60, H = 6, R = m = 10) | Primal block (ours) |

Everything before and after the cross-modal stage (temporal convolutions, the three
self-attention "memory" transformers, the head) is MulT's and is shared by the three arms.
With `fusion=crossmodal` the patched model is the unmodified MulT.

## Setup

```bash
# from the root of this repository
git clone https://github.com/thuiar/MMSA
git -C MMSA checkout a94e65d07fa1ae0d44e552390074b29b0898edfd
git -C MMSA apply ../experiments/mosei/mmsa.patch
pip install -r experiments/mosei/requirements.txt   # pinned versions, see below
pip install -e MMSA                                 # MMSA
pip install -e .                                    # primal_block
```

Install the pinned requirements first. MMSA's own dependency list is unpinned, and a fresh
resolve picks versions it does not run with (among them a `botocore` that fails to import
on Python 3.10 and later).

The `MMSA/` clone is listed in `.gitignore`.

Tested with Python 3.10 and 3.11 and the versions in `requirements.txt`.

## Data

Download `MOSEI/Processed/unaligned_50.pkl` (13.7 GB) from the links in the MMSA README
("Datasets"). SHA-256, as listed there and as used for the paper:
`ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f`. The file holds MMSA's
splits (16,326 / 1,871 / 4,659 clips) and unaligned features: 50 text tokens of frozen
BERT features, up to 500 audio frames and up to 375 vision frames.

## Run

Run from the root of this repository (not from inside the MMSA clone; `run.py` and
`report.py` live in `experiments/mosei/`). `--data-root` is the directory that contains
`MOSEI/Processed/unaligned_50.pkl`.

```bash
mkdir -p logs
for arm in mult mult_nofusion primal lmf; do
    python experiments/mosei/run.py $arm --data-root /path/to/MMSA-data --gpu 0 > logs/$arm.log 2>&1
done
python experiments/mosei/report.py logs
```

Each arm trains five seeds (1111 to 1115). `report.py` prints the Table 2 metrics as
mean ± SD and the paired t-tests against MulT.

The three MulT-skeleton arms use MulT's published MOSEI configuration from MMSA with one
change: the batch of 32 is processed in a single step (`batch_size=32`, `update_epochs=1`,
`grad_clip=0.075`), where MMSA's default accumulates 8 steps of batch 4 with a clipping
threshold of 0.6. LMF uses MMSA's configuration unchanged (batch 32).

## Stored logs

`results/mosei/logs/` holds the logs behind Table 2, one file per arm and seed (one file
with five seeds for LMF):

```bash
python experiments/mosei/report.py results/mosei/logs
```

They are the original stdout of the runs with progress bars removed and local paths
replaced by `<DATA>` and `<MMSA>`. Checkpoint paths inside them carry the working names the
arms had at the time (`fast_crossmodal` = MulT, `fast_nocross` = MulT w/o fusion,
`fast_primal_odd` = primal block).
