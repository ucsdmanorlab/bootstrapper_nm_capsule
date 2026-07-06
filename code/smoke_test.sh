#!/usr/bin/env bash
# smoke_test.sh - fast end-to-end pipeline sanity check for the sparsity capsule.
#
# WHAT THIS IS: a reproducibility pre-flight that runs EVERY stage of EVERY track
# (pgt, bootstrap, baseline) end to end, but small and fast. For each track it
# stages a throwaway copy into results/_smoke/, trains the net for only
# SMOKE_ITERS iterations, then runs the real 02_pred / 03_seg (+ post_fill) /
# 04_eval / 05_filter stages on a small predict ROI. So it exercises the daisy /
# gunpowder predict, the frozen 2D->3D corrector, MWS segmentation, hole-fill,
# NVI eval, and the pseudo-GT filter - the stages that a full run crashes in if a
# config, path, checkpoint, or library problem exists (the train-only smoke this
# replaced could not catch those; it passed while the full run died at predict).
#
# It preserves the trained checkpoint's iteration number in its filename (trains
# SMOKE_ITERS, copies model_checkpoint_<SMOKE_ITERS> onto model_checkpoint_20000/
# _30000) so every downstream path (chain_str, seg dirs, the pgt->bootstrap
# label dependency) stays valid without rewriting them.
#
# WHAT THIS IS NOT: a reproduction. The nets are undertrained and the ROI is a
# small block, so the segmentations and NVI numbers are meaningless. It answers
# only "does every stage run and produce its output artifact?". PASS/FAIL is
# decided per stage by checking that artifact exists (bs run swallows subprocess
# failures and exits 0, so exit codes are not trusted). For a real run:
#
#     ./code/run all          # every dataset, all tracks - hours
#
# USAGE:
#     ./code/smoke_test.sh                 # all 11 datasets, all their tracks
#     ./code/smoke_test.sh cremi_a epi     # only the named datasets
#     SMOKE_ITERS=50 ./code/smoke_test.sh cremi_a    # even faster (may degenerate)
#
# Exit status is 0 only if every stage of every checked track passed.

set -uo pipefail

# ---- config ---------------------------------------------------------------
SMOKE_ITERS="${SMOKE_ITERS:-200}"      # training iters per net (enough for a
                                       # non-degenerate seg, not convergence)
STAGE_TIMEOUT="${STAGE_TIMEOUT:-900}"  # seconds per stage; a hang is a failure
ALL_DATASETS="cremi_a cremi_b cremi_c epi fib harris15 liconn mitoem cremi_clefts fluo prism"

# ---- locate capsule root --------------------------------------------------
CAP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CAP"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Committed tomls carry absolute paths under the dev root; rewrite that prefix to
# THIS run's root when staging (matches code/run; a no-op on the dev box, and on
# Code Ocean maps to /code //data //results). Then redirect result paths into the
# smoke sandbox so real results/ is never touched.
DEV_ROOT="/data/data6/vijay/sparsity_capsule"
RUN_ROOT="${CAP%/}"
SANDBOX="$CAP/results/_smoke"

LOG="results/smoke_test.log"
mkdir -p results
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=================================================================="
echo " sparsity capsule smoke test (end-to-end plumbing, not a repro)"
echo " capsule:    $CAP"
echo " iterations: $SMOKE_ITERS per net    stage timeout: ${STAGE_TIMEOUT}s"
echo " sandbox:    $SANDBOX"
echo " log:        $LOG"
echo " started:    $(date)"
echo "=================================================================="

command -v bs >/dev/null || { echo "error: 'bs' not on PATH (activate the bootstrapper env)"; exit 1; }

if [ "$#" -gt 0 ]; then datasets="$*"; else datasets="$ALL_DATASETS"; fi

# ---- always clean up the sandbox, even on Ctrl-C / error ------------------
cleanup() {
  echo
  echo "----- removing smoke sandbox -----"
  rm -rf "$SANDBOX" 2>/dev/null && echo "  removed $SANDBOX"
}
trap cleanup EXIT INT TERM

# read a top-level string key from a toml
tomlget() { python -c "import toml,sys; print(toml.load(sys.argv[1]).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null; }

# a dir that exists and is non-empty (a zarr array dir, or a prefix of them)
have() { [ -e "$1" ] && { [ -f "$1" ] || [ -n "$(ls -A "$1" 2>/dev/null)" ]; }; }

declare -a ROWS   # "dataset|track|stage|RESULT"
record() { ROWS+=("$1|$2|$3|$4"); }

# ---- run every stage of one track ----------------------------------------
# returns 0 if the whole track passed (or the track does not exist), 1 on any FAIL
run_track() {
  local d="$1" t="$2"
  local src="code/setups/$d/$t"
  local dst="$SANDBOX/$d/$t"
  [ -d "$src" ] || return 0    # track not present for this dataset

  echo
  echo "================= $d / $t ================="
  rm -rf "$dst"; mkdir -p "$SANDBOX/$d"; cp -r "$src" "$dst"

  # 1) portability rewrite (matches code/run), 2) redirect results into sandbox
  find "$dst" -name '*.toml' -print0 | xargs -0 sed -i "s#${DEV_ROOT}#${RUN_ROOT}#g"
  find "$dst" -name '*.toml' -print0 | xargs -0 sed -i "s#${RUN_ROOT}/results/#${RUN_ROOT}/results/_smoke/#g"

  # 3) cut iters, shrink predict ROI, realpath setup_dir; capture checkpoint copy
  local prep ckpt_src ckpt_dst
  prep="$(python code/smoke_prep.py "$CAP" "$dst" "$d" "$t" "$SMOKE_ITERS")"
  echo "$prep"
  ckpt_src="$(awk '/^CKPT_COPY/{print $2}' <<<"$prep")"
  ckpt_dst="$(awk '/^CKPT_COPY/{print $3}' <<<"$prep")"

  local ok=1
  local toml stage rc segp affs outr outs
  for toml in "$dst"/run/*.toml; do
    stage="$(basename "$toml")"
    echo "----- bs run $stage -----"
    timeout "$STAGE_TIMEOUT" bs run "$toml"
    rc=$?
    if [ "$rc" -eq 124 ]; then
      echo "  RESULT: TIMEOUT ($stage)"; record "$d" "$t" "$stage" TIMEOUT; ok=0; break
    fi
    case "$toml" in
      *01_train*)
        if [ -z "$ckpt_src" ] || [ -z "$ckpt_dst" ]; then
          echo "  FAIL train: smoke_prep emitted no CKPT_COPY (see prep output above)"; record "$d" "$t" train FAIL; ok=0; break
        fi
        if [ ! -f "$ckpt_src" ]; then
          echo "  FAIL train: checkpoint $ckpt_src not written (training crashed?)"; record "$d" "$t" train FAIL; ok=0; break
        fi
        cp -f "$ckpt_src" "$ckpt_dst"
        if [ ! -f "$ckpt_dst" ]; then
          echo "  FAIL train: could not place checkpoint at $ckpt_dst"; record "$d" "$t" train FAIL; ok=0; break
        fi
        echo "  train PASS ($(basename "$ckpt_src") -> $(basename "$ckpt_dst"))"; record "$d" "$t" train PASS ;;
      *02_pred*)
        affs="$(tomlget "$dst/run/03_seg_$d.toml" affs_dataset)"
        if have "$affs/.zarray" || have "$affs"; then
          echo "  pred PASS"; record "$d" "$t" pred PASS
        else
          echo "  FAIL pred: affinities not written ($affs)"; record "$d" "$t" pred FAIL; ok=0; break
        fi ;;
      *03_seg*)
        python code/post_fill.py "$toml"
        segp="$(tomlget "$toml" seg_dataset_prefix)"
        if have "$segp"; then echo "  seg PASS"; record "$d" "$t" seg PASS
        else echo "  FAIL seg: no segmentation under $segp"; record "$d" "$t" seg FAIL; ok=0; break; fi ;;
      *04_eval*)
        outr="$(tomlget "$toml" out_result)"
        if have "$outr"; then echo "  eval PASS"; record "$d" "$t" eval PASS
        else echo "  FAIL eval: no result json $outr"; record "$d" "$t" eval FAIL; ok=0; break; fi ;;
      *05_filter*)
        outs="$(tomlget "$toml" out_seg_dataset_prefix)"
        if have "$outs"; then echo "  filter PASS"; record "$d" "$t" filter PASS
        else echo "  FAIL filter: no pseudo-GT under $outs"; record "$d" "$t" filter FAIL; ok=0; break; fi ;;
    esac
  done
  [ "$ok" -eq 1 ]
}

# ---- main loop: pgt first (bootstrap trains on its output), then baseline -
for d in $datasets; do
  [ -d "code/setups/$d" ] || { echo "skip $d: no setup at code/setups/$d"; continue; }
  if run_track "$d" pgt; then pgt_ok=1; else pgt_ok=0; fi
  if [ -d "code/setups/$d/bootstrap" ]; then
    if [ "$pgt_ok" -eq 1 ]; then run_track "$d" bootstrap
    else echo; echo "  SKIP $d/bootstrap: pgt failed (bootstrap trains on pgt output)"; record "$d" bootstrap "(deps)" SKIP; fi
  fi
  run_track "$d" baseline
done

# ---- summary --------------------------------------------------------------
echo
echo "=================================================================="
echo " SMOKE TEST SUMMARY (end-to-end, $SMOKE_ITERS iters, small ROI)"
echo "=================================================================="
printf " %-14s %-10s %-16s %s\n" DATASET TRACK STAGE RESULT
printf " %-14s %-10s %-16s %s\n" -------------- ---------- ---------------- ------
fails=0
for row in "${ROWS[@]:-}"; do
  [ -z "$row" ] && continue
  IFS='|' read -r rd rt rs rr <<<"$row"
  printf " %-14s %-10s %-16s %s\n" "$rd" "$rt" "$rs" "$rr"
  [ "$rr" = PASS ] || fails=$((fails + 1))
done
echo "------------------------------------------------------------------"
if [ "$fails" -eq 0 ]; then echo " ALL STAGES PASSED"; else echo " $fails stage(s) FAILED/SKIPPED - see log above"; fi
echo " finished: $(date)"
echo " full log: $LOG"
echo "=================================================================="

[ "$fails" -eq 0 ]
