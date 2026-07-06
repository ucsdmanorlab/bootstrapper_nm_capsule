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


def labelled_locations(labels_path, mask_path=None, max_points=4000, seed=42):
    """World-coordinate (z,y,x in nm) points at labelled voxels, for use with
    gp.SpecifiedLocation. Centering a request on one of these lands annotated
    voxels in the output ROI, which is what makes training on sparse annotation
    tractable: gp.RandomLocation either hangs building a mask integral over a
    large sparse array (epi/mito) or reject-spams when labels occupy a tiny
    fraction of z (fib). We know where the labels are, so we sample them.

    Points are at voxel centers so the single-section (z=1) output ROI lands
    exactly on the annotated section. We center on label objects (labels>0)
    restricted to the valid mask: for per-voxel sparse (harris/liconn/epi) the
    mask already equals labels>0, while for whole-section masks (mito, clefts,
    fluo) this picks the actual objects inside the section instead of sampling
    background, so every crop carries positive labels.
    """
    lab = open_ds(labels_path)
    fg = lab[lab.roi] > 0
    if mask_path is not None and os.path.isdir(mask_path):
        m = open_ds(mask_path)
        if m.roi.contains(lab.roi):
            md = m[lab.roi] > 0
            if md.shape == fg.shape:
                fg = fg & md
    idx = np.argwhere(fg)
    if idx.shape[0] == 0:
        raise RuntimeError(f"no labelled voxels found in {labels_path}")
    rng = np.random.default_rng(seed)
    if idx.shape[0] > max_points:
        idx = idx[rng.choice(idx.shape[0], max_points, replace=False)]
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
                labelled_locations(sample["labels"], mask_path),
                choose_randomly=True,
            )
        )
        sources.append(src)

    pipeline = tuple(sources) + gp.RandomProvider()

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])
    pipeline += gp.DeformAugment(
        control_point_spacing=gp.Coordinate((voxel_size[-2] * voxel_size[0], voxel_size[-1] * voxel_size[0])),
        jitter_sigma=(2.0 * voxel_size[-2], 2.0 * voxel_size[-1]),
        spatial_dims=2,
        subsample=1,
        scale_interval=(0.9, 1.1),
        p=0.5,
    )
    if adj_slices > 1 and section_augment:
        pipeline += gp.ShiftAugment(prob_slip=0.2, prob_shift=0.2, sigma=3)
    pipeline += gp.NoiseAugment(raw, p=0.5)
    pipeline += gp.IntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.1,
        shift_min=-0.1,
        shift_max=0.1,
        slab=(1, -1, -1),
        p=0.5,
    )
    pipeline += GammaAugment(raw, slab=(1, -1, -1), p=0.5)
    pipeline += ImpulseNoiseAugment(raw, pixel_p=0.05, p=0.5)
    pipeline += SmoothAugment(raw, p=0.5)
    pipeline += DefectAugment(raw, prob_missing=0.05 if (adj_slices > 1 and section_augment) else 0.0, prob_low_contrast=0.1)

    pipeline += Add2DLSDs(
        labels,
        gt_lsds,
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
        labels_mask=unlabelled,
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
