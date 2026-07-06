import os
import sys
# Make `from model import ...` resolve when this file is re-imported by a
# forkserver/spawn worker, whose sys.path[0] is not this script's directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import gunpowder as gp
from funlib.persistence import open_ds
from model import Model, WeightedMSELoss

import toml
import json
import logging
import numpy as np

from bootstrapper.gp import (
    SmoothAugment,
    Add2DLSDs,
    CreateMask,
    Renumber,
    calc_max_padding,
    DefectAugment, 
    GammaAugment, 
    ImpulseNoiseAugment
)


logging.getLogger().setLevel(logging.INFO)
setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

torch.backends.cudnn.benchmark = True


def labelled_locations(labels_path, mask_path=None, max_points=5000, seed=42, bg_ratio=0.7):
    """World-coordinate (z, y, x nm) voxel-center points for gp.SpecifiedLocation.

    Foreground points are at (labels > 0) & (mask > 0), so a request centered
    there lands annotated objects in the output ROI. When a whole-section mask is
    given, a fraction bg_ratio of the points are drawn from in-section background
    (mask > 0) & (labels == 0) so the net also sees non-object crops. For
    per-voxel-sparse data (mask == labels > 0) that set is empty and bg_ratio is
    a no-op.
    """
    lab = open_ds(labels_path)
    labd = lab[lab.roi]
    fg = labd > 0
    md = None
    if mask_path is not None and os.path.isdir(mask_path):
        m = open_ds(mask_path)
        if m.roi.contains(lab.roi):
            mm = m[lab.roi] > 0
            if mm.shape == fg.shape:
                md = mm
                fg = fg & md
    fg_idx = np.argwhere(fg)
    if fg_idx.shape[0] == 0:
        raise RuntimeError(f"no labelled voxels found in {labels_path}")

    # in-section background: mask>0 AND labels==0. Empty (-> no-op) whenever
    # mask==labels>0 (per-voxel-sparse) or when there is no separate mask.
    if md is not None:
        bg = md & (labd == 0)
        bg_idx = np.argwhere(bg)
    else:
        bg_idx = np.zeros((0, fg_idx.shape[1]), dtype=fg_idx.dtype)

    rng = np.random.default_rng(seed)

    def _subsample(idx, n):
        if n <= 0 or idx.shape[0] == 0:
            return idx[:0]
        if idx.shape[0] > n:
            idx = idx[rng.choice(idx.shape[0], n, replace=False)]
        return idx

    if bg_idx.shape[0] > 0 and bg_ratio > 0:
        n_bg = int(round(max_points * bg_ratio))
        n_fg = max_points - n_bg
        fg_idx = _subsample(fg_idx, n_fg)
        bg_idx = _subsample(bg_idx, n_bg)
        idx = np.concatenate([fg_idx, bg_idx], axis=0)
    else:
        idx = _subsample(fg_idx, max_points)

    off = np.array(lab.roi.offset)
    vs = np.array(lab.voxel_size)
    half = vs // 2
    return [tuple(int(c) for c in (off + i * vs + half)) for i in idx]


def train(
    setup_dir,
    voxel_size,
    max_iterations,
    samples,
    save_checkpoints_every,
    save_snapshots_every,
):
    # array keys
    raw = gp.ArrayKey("RAW")
    labels = gp.ArrayKey("LABELS")
    unlabelled = gp.ArrayKey("UNLABELLED")

    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")

    gt_affs = gp.ArrayKey("GT_AFFS")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    gt_affs_mask = gp.ArrayKey("AFFS_MASK")
    pred_affs = gp.ArrayKey("PRED_AFFS")

    # model training setup
    model = Model(stack_infer=True)
    model.train()
    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    batch_size = 10

    with open(os.path.join(setup_dir, "net_config.json")) as f:
        logging.info(
            "Reading setup config from %s" % os.path.join(setup_dir, "net_config.json")
        )
        net_config = json.load(f)

    # get affs task params
    neighborhood = net_config["outputs"]["2d_affs"]["neighborhood"]
    neighborhood = [
        [0, *x] for x in neighborhood
    ]  # add z-dimension since pipeline is 3D
    aff_grow_boundary = net_config["outputs"]["2d_affs"]["grow_boundary"]

    # get lsd task params
    sigma_vox = net_config["outputs"]["2d_lsds"]["sigma"]
    # sigma is specified in VOXELS; convert to world units (nm) per axis so the LSD
    # context is a fixed voxel count across datasets and matches the pretrained
    # corrector (world-unit sigma would vary wildly per voxel size: 80 nm is ~10
    # voxels at harris 8 nm but ~1 voxel at epi 75 nm).
    sigma = (0, sigma_vox * voxel_size[-2], sigma_vox * voxel_size[-1])
    lsd_downsample = net_config["outputs"]["2d_lsds"]["downsample"]
    
    adj_slices = net_config["adj_slices"]
    section_augment = net_config.get("section_augment", True)
    shape_increase = [0, 0]
    input_shape = [x + y for x, y in zip(shape_increase, net_config["input_shape"])]
    output_shape = [x + y for x, y in zip(shape_increase, net_config["output_shape"])]

    # prepare request
    voxel_size = gp.Coordinate(voxel_size)
    input_size = gp.Coordinate((adj_slices, *input_shape)) * voxel_size
    output_size = gp.Coordinate((1, *output_shape)) * voxel_size

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(labels, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    # prepare pipeline: one source per sample, sampled at known labelled
    # locations (gp.SpecifiedLocation) rather than gp.RandomLocation/gp.Reject,
    # which hang or reject-spam on sparse annotation. unlabelled is padded too so
    # near-edge labelled locations are never skipped (pad value 0 == unlabelled).
    sources = []
    for sample in samples:
        mask_path = sample.get("mask")
        if mask_path is not None:
            src = (
                gp.ArraySource(raw, open_ds(sample["raw"]), True),
                gp.ArraySource(labels, open_ds(sample["labels"]), False),
                gp.ArraySource(unlabelled, open_ds(mask_path), False),
            ) + gp.MergeProvider()
        else:
            src = (
                gp.ArraySource(raw, open_ds(sample["raw"]), True),
                gp.ArraySource(labels, open_ds(sample["labels"]), False),
            ) + gp.MergeProvider() + CreateMask(labels, unlabelled)

        src += (
            gp.Normalize(raw)
            + Renumber(labels)
            + gp.AsType(labels, "uint32")
            + gp.Pad(raw, None)
            + gp.Pad(labels, None)
            + gp.Pad(unlabelled, None)
            + gp.SpecifiedLocation(
                labelled_locations(sample["labels"], mask_path, bg_ratio=0.7),
                choose_randomly=True,
            )
        )
        sources.append(src)

    pipeline = tuple(sources) + gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])
    pipeline += gp.DeformAugment(
        control_point_spacing=gp.Coordinate((voxel_size[-2] * voxel_size[0], voxel_size[-1] * voxel_size[0])),
        jitter_sigma=(1.0 * voxel_size[-2], 1.0 * voxel_size[-1]),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )
    if adj_slices > 1 and section_augment:
        pipeline += gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=3)

    pipeline += Add2DLSDs(
        labels,
        gt_lsds,
        unlabelled=unlabelled,
        labels_mask=unlabelled,
        lsds_mask=lsds_weights,
        sigma=sigma,
        downsample=lsd_downsample,
    )

    pipeline += gp.GrowBoundary(labels, mask=unlabelled, steps=aff_grow_boundary, only_xy=True)

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        affinities=gt_affs,
        unlabelled=unlabelled,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(gt_affs, affs_weights, mask=gt_affs_mask)

    pipeline += gp.IntensityScaleShift(raw, 2, -1)

    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(cache_size=32, num_workers=16)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={0: raw},
        loss_inputs={
            0: pred_lsds,
            1: gt_lsds,
            2: lsds_weights,
            3: pred_affs,
            4: gt_affs,
            5: affs_weights,
        },
        outputs={
            0: pred_lsds,
            1: pred_affs,
        },
        log_dir=os.path.join(setup_dir, "log"),
        checkpoint_basename=os.path.join(setup_dir, "model"),
        save_every=save_checkpoints_every,
    )

    pipeline += gp.IntensityScaleShift(raw, 0.5, 0.5)

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
            gt_lsds: "gt_lsds",
            pred_lsds: "pred_lsds",
            lsds_weights: "lsds_weights",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
            affs_weights: "affs_weights",
        },
        output_filename="batch_{iteration}.zarr",
        output_dir=os.path.join(setup_dir, "snapshots"),
        every=save_snapshots_every,
    )


    with gp.build(pipeline):
        for i in range(max_iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":

    import multiprocessing as mp
    # Spawn precache workers via forkserver (a CUDA-free server process) instead
    # of fork. The Train node initializes CUDA before PreCache forks its data
    # workers; forking after CUDA init deadlocks those workers (they wedge on a
    # futex at 0% CPU and never produce a batch) when several trainings run at
    # once. forkserver workers never inherit the parent's CUDA state.
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    config_file = sys.argv[1]
    with open(config_file, "r") as f:
        config = toml.load(f)

    assert config["setup_dir"] in setup_dir, "model directories do not match"
    config["setup_dir"] = setup_dir

    train(**config)
