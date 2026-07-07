# Bootstrapper NM capsule

Reproduction of "A general method for bootstrapping dense 3D segmentations from
sparse 2D annotations". The capsule turns sparse 2D
annotation into a dense 3D segmentation that can be used as pseudo ground-truth, and shows that a 3D model bootstrapped from
that pseudo ground-truth approaches one trained on dense ground truth.

This capsule uses [bootstrapper](github.com/ucsdmanorlab/bootstrapper) to drive the runs. 

## Core datasets

Every core dataset is split into two volumes. 

- **volume_1** carries the sparse 2D annotations.
- **volume_2** is held out.

Three tracks, scored against dense ground truth with NVI (lower is better):

1. **pgt** (pseudo ground truth): a `2d_mtlsd` (multi-task local shape descriptors) net trains on volume_1's sparse
   annotation, is chained through a pretrained **2D->3D net**
   (one of the pretrained `3d_affs_from_2d_*` correctors in `code/nets/`), and is tested
   on the held-out **volume_2**. This sparse-2D-to-3D result becomes the bootstrap's
   training volume.
2. **bootstrap**: a `3d_mtlsd` net trains on volume_2's pgt, then is tested on the
   held-out **volume_1**.
3. **baseline**: a `3d_mtlsd` net trains on volume_2's dense ground truth, scored on
   the held-out **volume_1**. `  

A run writes the held-out NVI numbers to `results/summary.md`.

> The Code Ocean reproducible run executes the **pgt** track only (see Running).
> Use `./code/run --full ...`, or run the full pipeline from the GitHub repo, to
> also train the bootstrap and baseline 3D models.

## Datasets

Six datasets have two volumes and run all three tracks: **cremi_a, cremi_b,
cremi_c, epi, fib, harris15**. Five are single-volume extras that run pgt only:
**liconn, mitoem, cremi_clefts, fluo, prism**.

| dataset | modality | voxel (z,y,x nm) | tracks |
|---|---|---|---|
| cremi_a, cremi_b, cremi_c | EM, Drosophila neuropil | 40, 8, 8 | all three |
| epi | plant epithelium | 235, 75, 75 | all three |
| fib | FIB-SEM, isotropic | 8, 8, 8 | all three |
| harris15 | EM, hippocampal neuropil | 50, 8, 8 | all three |
| liconn | expansion-microscopy LM | 24, 18, 18 | pgt |
| mitoem | EM, mitochondria | 30, 8, 8 | pgt |
| cremi_clefts | EM, synaptic clefts | 40, 4, 4 | pgt |
| fluo | fluorescence, 2D+time | 1, 1, 1 | pgt |
| prism | PRISM expansion LM, 18-channel | 400, 168, 168 | pgt |

## Setup

The code and the pretrained 2D->3D corrector checkpoints (stored with
[Git LFS](https://git-lfs.com)) live in this repo. The dataset volumes (~3.9 GB)
are archived separately on [Zenodo](https://zenodo.org/records/21223591) and
fetched with a script:

```bash
git clone https://github.com/ucsdmanorlab/bootstrapper_nm_capsule.git
cd bootstrapper_nm_capsule
git lfs pull          # fetch the corrector checkpoints (code/nets/3d_affs_from_2d_*/)
./download_data.sh    # download data.tar from Zenodo and unpack it into data/
```

Build the environment from `environment/` (pinned `requirements.txt`) and activate
it so the `bs` CLI is on PATH. Optional fast end-to-end check of every stage and
track before a full run:

```bash
./code/smoke_test.sh cremi_a
```

## Running

```bash
./code/run               # cremi_a, pgt track only (the quick demo)
./code/run all           # every dataset, pgt track only
./code/run cremi_c epi   # named datasets, pgt track only
```

The reproducible run does only the pgt track: the sparse-2D-to-3D
pseudo-ground-truth step, which is the method's novel contribution. The 3D
bootstrap and baseline models can train slowly, so they are left out of the default
run.

The bootstrap and baseline setups are still in the capsule
(code/setups/<dataset>/{bootstrap,baseline}/). To run the full sequence (pgt,
then bootstrap trained on the pgt, then baseline on dense ground truth), add
--full:

`./code/run --full all`        # every dataset, all three tracks
`./code/run --full cremi_a`    # one dataset, all three tracks

The driver stages each setup into results/ and runs its stages in order. Each
invocation re-stages and re-runs from scratch. When it finishes it writes
results/summary.md and results/summary.json. The pgt column is always filled;
the bootstrap/baseline columns and their gap appear only under `--full`.


## Layout

```
code/
  run                     driver (the only orchestration; plain shell, no config generation)
  collect_results.py      reads the run's eval JSONs into results/summary.{md,json}
  smoke_test.sh           fast end-to-end sanity check (every stage, every track)
  nets/                   pretrained 2D->3D correctors (recipe + one checkpoint each, via Git LFS):
    3d_affs_from_2d_{aff,aff_6ch,lsd_epi,lsd_fine,mtlsd}/
  setups/<dataset>/<track>/   one self-contained `bs run` setup
    {2d_mtlsd|3d_mtlsd}/   the net recipe (model, unet, net_config, train, predict)
    run/                   01_train, 02_pred, 03_seg, 04_eval, (05_filter for pgt)
download_data.sh          fetch + unpack the volumes from Zenodo into data/
data/<dataset>/volume_{1,2}.zarr/{raw,labels,sparse_labels}   the volumes (Zenodo; /data on Code Ocean)
environment/              Dockerfile + pinned requirements
results/                  run outputs land here; a run writes summary.{md,json}
metadata/metadata.yml     Code Ocean metadata
```

To inspect a setup, open `code/setups/<dataset>/<track>/run/`.

The committed tomls hold absolute paths under the dev capsule root; `code/run`
rewrites that prefix to the current root when it stages each setup (so the same
configs work locally and on Code Ocean, where code/data/results mount at
`/code`, `/data`, `/results`). The simplest way to run anything is the driver:

```bash
./code/run cremi_a          # one dataset, all its tracks
```

To run a single stage by hand you must apply the same rewrite the driver does,
or the staged toml will point at paths that do not exist here:

```bash
cp -r code/setups/cremi_a/pgt results/cremi_a/pgt
ROOT="$(cd "$(dirname code/run)/.." && pwd)"   # this capsule's root
find results/cremi_a/pgt -name '*.toml' -print0 \
  | xargs -0 sed -i "s#/data/data6/vijay/sparsity_capsule#${ROOT%/}#g"
bs run results/cremi_a/pgt/run/01_train_00.toml
```

Each track segments once, at the operating point that was best for that dataset
(baked into `03_seg_<dataset>.toml`). 

## Data

The dataset volumes are **not** stored in git. They are publicly available,
consolidated into a single archive (`data.tar`, ~3.9 GB), and hosted on
[Zenodo](https://zenodo.org/records/21223591). `./download_data.sh` downloads it
and unpacks it into `data/`. On Code Ocean the same volumes are attached as a
Data Asset mounted at `/data`, so the script is not needed there.

Once unpacked, each volume is a zarr array: core datasets have
`volume_1.zarr/{raw,labels,sparse_labels}` and `volume_2.zarr/{raw,labels}`;
pgt-only datasets have only `volume_1.zarr` (plus an added `sparse_labels_mask`).




