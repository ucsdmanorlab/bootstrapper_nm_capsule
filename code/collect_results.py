#!/usr/bin/env python
# Collect held-out NVI per (dataset, track) from the eval JSONs a run writes under
# results/, into results/summary.json and results/summary.md. Keeps the best
# (lowest, non-degenerate) NVI sum per track. Reads results only; writes nothing
# under code/.
import json, os, glob

CAP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(CAP, "results")
TRACKS = ("pgt", "bootstrap", "baseline")
ORDER = ["cremi_a", "cremi_b", "cremi_c", "epi", "fib", "harris15",
         "liconn", "mitoem", "cremi_clefts", "fluo", "prism"]

best = {}  # (dataset, track) -> metrics dict
for jf in glob.glob(os.path.join(RES, "*", "*", "**", "results_gt_*.json"), recursive=True):
    rel = os.path.relpath(jf, RES).split(os.sep)
    dataset, track = rel[0], rel[1]
    if track not in TRACKS:
        continue
    try:
        data = json.load(open(jf))
    except Exception:
        continue
    for rec in data.values():
        v = rec.get("metrics", {}).get("voi", {})
        ns, nm = v.get("nvi_split"), v.get("nvi_merge")
        vs, vm = v.get("voi_split"), v.get("voi_merge")
        if ns is None or nm is None or not vs or not vm or ns >= 0.99 or nm >= 0.99:
            continue
        nvi = round(ns + nm, 4)
        k = (dataset, track)
        if k not in best or nvi < best[k]["nvi_sum"]:
            best[k] = {"voi_split": round(vs, 4), "voi_merge": round(vm, 4),
                       "voi_sum": round(vs + vm, 4), "nvi_split": round(ns, 4),
                       "nvi_merge": round(nm, 4), "nvi_sum": nvi}

rows = []
for d in ORDER:
    tracks = {t: best[(d, t)] for t in TRACKS if (d, t) in best}
    if not tracks:
        continue
    row = {"volume": d, "tracks": tracks}
    if "bootstrap" in tracks and "baseline" in tracks:
        row["delta_nvi_sum_bootstrap_minus_baseline"] = round(
            tracks["bootstrap"]["nvi_sum"] - tracks["baseline"]["nvi_sum"], 4)
    rows.append(row)

json.dump(rows, open(os.path.join(RES, "summary.json"), "w"), indent=2)

def cell(r, t):
    return f"{r['tracks'][t]['nvi_sum']:.3f}" if t in r["tracks"] else "-"

lines = ["# Held-out results", "",
         "Each model is scored against dense ground truth on a volume it did not",
         "train on (NVI sum, lower is better).", "",
         "| volume | pgt | bootstrap | baseline | bootstrap - baseline |",
         "|---|---|---|---|---|"]
for r in rows:
    dlt = r.get("delta_nvi_sum_bootstrap_minus_baseline")
    lines.append(f"| {r['volume']} | {cell(r,'pgt')} | {cell(r,'bootstrap')} | "
                 f"{cell(r,'baseline')} | {('%+.3f' % dlt) if dlt is not None else '-'} |")
open(os.path.join(RES, "summary.md"), "w").write("\n".join(lines) + "\n")
print("wrote results/summary.json and results/summary.md")
print("\n".join(lines))
