"""Post-process a segmentation: fill interior holes per section (fastmorph).

Runs after 03_seg and before 04_eval (see code/run). Reads the seg dataset(s) that
03_seg wrote and fills holes in place, so the evaluated segmentation is the filled one.
Filling interior holes lowers NVI on every dataset tested. Optional size filter
(fill_size_filter in the seg toml) drops sub-threshold objects to background first,
which merges background fragments into a single id.
"""
import sys
import glob
import numpy as np
import toml
import fastmorph
from funlib.persistence import open_ds


def _size_filter(seg, thr):
    ids, cnt = np.unique(seg, return_counts=True)
    small = ids[(cnt < thr) & (ids != 0)]
    if small.size:
        seg[np.isin(seg, small)] = 0
    return seg


def _fill(seg):
    out = seg.copy()
    for z in range(out.shape[0]):
        blk = np.repeat(out[z:z + 1], 3, axis=0)
        blk = fastmorph.fill_holes_v1(blk, remove_enclosed=True, fix_borders=True, morphological_closing=True)
        blk = fastmorph.fill_holes_v2(blk, merge_threshold=0.95, fix_borders=True, parallel=2)[0]
        out[z] = blk[1]
    return out


def main(seg_toml):
    cfg = toml.load(seg_toml)
    if not cfg.get("fill_holes", False):
        return  # opt-in: only fill segs whose 03_seg toml sets fill_holes = true
    prefix = cfg["seg_dataset_prefix"]
    thr = int(cfg.get("fill_size_filter", 0))
    seg_dirs = [p for p in glob.glob(f"{prefix}/*") if not p.endswith(".json")]
    for sd in seg_dirs:
        ds = open_ds(sd, mode="a")
        data = ds[ds.roi]
        if thr > 0:
            data = _size_filter(data, thr)
        data = _fill(data)
        ds[ds.roi] = data
        print(f"filled {sd}" + (f" (size_filter {thr})" if thr else ""))


if __name__ == "__main__":
    main(sys.argv[1])
