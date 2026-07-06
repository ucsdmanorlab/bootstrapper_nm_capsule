#!/usr/bin/env python
"""Prepare a staged track's run/ tomls for a fast end-to-end smoke.

Called by smoke_test.sh AFTER a track has been staged into the smoke sandbox
(results/_smoke/<dataset>/<track>) and its paths rewritten to that sandbox.

It does the three edits a smoke needs that the driver does not:

  1. 01_train: cut max_iterations / save_checkpoints_every to SMOKE_ITERS, and
     resolve setup_dir through symlinks (train.py asserts setup_dir is a
     substring of realpath(__file__); results/ is a symlink on Code Ocean).
  2. 02_pred: shrink the predict ROI to a small block so predict/seg/filter run
     in seconds. The block is sized from the net input shapes and the data
     volume. For a pgt track whose dataset also has a bootstrap track, the block
     is grown so the pgt segmentation is large enough for the 3D net's
     SpecifiedLocation sampler to find interior labelled voxels (bootstrap trains
     on the pgt output).
  3. print a "CKPT_COPY <src> <dst>" line so the caller can copy the smoke
     checkpoint (model_checkpoint_<SMOKE_ITERS>) onto the name the rest of the
     pipeline expects (model_checkpoint_20000 / _30000). Preserving the iteration
     number in the name keeps every downstream path (chain_str, seg dirs) valid
     without rewriting them.

Only 01_train and 02_pred are touched; 03_seg/04_eval/05_filter run unchanged on
the shrunk ROI.
"""
import glob
import json
import os
import sys

import toml
from funlib.persistence import open_ds


def net_input_zyx(setup_dir):
    """Return the net's input shape as (z, y, x) voxels, or None."""
    cfg_path = os.path.join(setup_dir, "net_config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path) as f:
        nc = json.load(f)
    ishape = nc.get("input_shape")
    if ishape is None:
        return None
    if len(ishape) == 2:  # 2D net: prepend the z stack depth
        z = nc.get("adj_slices", 1)
        return (z, ishape[0], ishape[1])
    return tuple(ishape)


def main():
    cap, track_dir, dataset, track, smoke_iters = (
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    )

    tomls = sorted(glob.glob(os.path.join(track_dir, "run", "*.toml")))

    train_setup_dir = None
    for tp in tomls:
        cfg = toml.load(tp)
        if "samples" in cfg:  # 01_train
            cfg["max_iterations"] = smoke_iters
            cfg["save_checkpoints_every"] = smoke_iters
            cfg["save_snapshots_every"] = smoke_iters
            sd = cfg.get("setup_dir")
            if sd and os.path.exists(sd):
                cfg["setup_dir"] = os.path.realpath(sd)
            train_setup_dir = cfg["setup_dir"]
            with open(tp, "w") as f:
                toml.dump(cfg, f)

    # sibling bootstrap 3D net input shape governs the pgt sampling floor
    floor_zyx = None
    if track == "pgt":
        sib = os.path.join(cap, "code", "setups", dataset, "bootstrap", "3d_mtlsd")
        floor_zyx = net_input_zyx(sib)

    ckpt_copy = None
    for tp in tomls:
        cfg = toml.load(tp)
        sections = [k for k, v in cfg.items()
                    if isinstance(v, dict) and "chain_str" in v]
        if not sections:
            continue  # not a predict toml

        # predict volume = first section's first input dataset
        first = cfg[sections[0]]
        in_ds = open_ds(first["input_datasets"][0], "r")
        vs = tuple(in_ds.voxel_size)
        data_off_vox = tuple(o // v for o, v in zip(in_ds.roi.offset, vs))
        data_shape_vox = tuple(in_ds.roi.shape[i] // vs[i] for i in range(len(vs)))

        # collect net input blocks across the chain
        blocks = []
        for s in sections:
            b = net_input_zyx(cfg[s].get("setup_dir", ""))
            if b:
                blocks.append(b)
        if floor_zyx:
            blocks.append(floor_zyx)
        base = tuple(max(b[d] for b in blocks) for d in range(3)) if blocks \
            else (16, 128, 128)

        if floor_zyx:  # pgt feeding a bootstrap: leave room for interior sampling
            target = tuple(base[d] + 44 for d in range(3))
        else:
            target = tuple(base[d] + 16 for d in range(3))
        target = tuple(min(target[d], data_shape_vox[d]) for d in range(3))

        start_vox = tuple(data_off_vox[d] + (data_shape_vox[d] - target[d]) // 2
                          for d in range(3))
        roi_offset = [int(start_vox[d] * vs[d]) for d in range(3)]
        roi_shape = [int(target[d] * vs[d]) for d in range(3)]

        for s in sections:
            if "roi_shape" in cfg[s]:
                cfg[s]["roi_offset"] = roi_offset
                cfg[s]["roi_shape"] = roi_shape
            # the trained net's section tells us the checkpoint name to preserve.
            # Compare via realpath: on Code Ocean results/ is a symlink, so the
            # 01_train setup_dir was resolved to its realpath while this section's
            # setup_dir is still the logical path -- a plain == would never match.
            sd_s = cfg[s].get("setup_dir")
            if sd_s and train_setup_dir and \
                    os.path.realpath(sd_s) == os.path.realpath(train_setup_dir):
                dst = cfg[s]["checkpoint"]
                src = os.path.join(train_setup_dir, f"model_checkpoint_{smoke_iters}")
                ckpt_copy = (src, dst)

        with open(tp, "w") as f:
            toml.dump(cfg, f)

        print(f"ROI {dataset}/{track}: offset={roi_offset} shape={roi_shape} "
              f"(voxels {list(target)} of {list(data_shape_vox)})")

    if ckpt_copy:
        print(f"CKPT_COPY {ckpt_copy[0]} {ckpt_copy[1]}")
    elif train_setup_dir:
        # a train toml exists but no predict section pointed back at it -- the
        # caller cannot preserve the checkpoint name, so say so loudly
        print(f"CKPT_COPY_ERROR no predict section matched train setup_dir "
              f"{train_setup_dir}")


if __name__ == "__main__":
    main()
