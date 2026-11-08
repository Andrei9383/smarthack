#!/usr/bin/env bash
# Recover finished lab run outputs that never left their GPU worker (#202: the lab has no artifact download and the
# workers are not reachable by ssh). Submit one grid; each run lands on whatever GPU is free:
#
#   lab submit <card> --run train --cmd 'bash scripts/lab_fetch_outputs.sh {job}' \
#       --grid job=load@s0,load@s1,load@s2,load@s3,pv_fc,windnbm,windfailure
#
# 1. Every target run directory found on this worker (<work>/outputs/<id>, where the worker moves .lab-out after a
#    run) is printed to the log as tar.gz | base64 between "=====BEGIN LABDUMP <label>=====" and
#    "=====END LABDUMP <label>=====", after one "LABDUMP-SHA <label> <sha256> <file>" line per file.
# 2. If the job's own original run is not on this worker, the job is retrained with that run's exact command and its
#    new outputs ($LAB_OUTPUT_DIR) are printed the same way (label new-<job>-r<run>).
# Decode locally: python scripts/lab_decode_dump.py <lab logs file> <dest dir>.
# Skipped: checkpoints and caches (*.pt, *.parquet, *.npz, ...), files >= 12 MB, the worker's env file.
set -uo pipefail
job="${1:?usage: lab_fetch_outputs.sh <job>}"
TARGETS="${LAB_FETCH_TARGETS:-674 675 680 681 682 683 684 778}"
WORK="${LAB_WORK_DIR:-$(cd ../.. && pwd)}"
OUT="$WORK/outputs"
PY=.venv/bin/python

dump() {  # dump <label> <dir>
  local label="$1" dir="$2" files
  echo "== dump $label from $dir on $(hostname)"
  ls -la "$dir"
  files=$(cd "$dir" && find . -maxdepth 2 -type f -size -12M ! -name env ! -name '*.pt' ! -name '*.pth' \
      ! -name '*.ckpt' ! -name '*.parquet' ! -name '*.npz' ! -name '*.npy' ! -name '*.pkl' ! -name '*.sqlite' \
      ! -name '*.png' | sort)
  (cd "$dir" && echo "$files" | xargs -r sha256sum) | sed "s/^/LABDUMP-SHA $label /"
  echo "=====BEGIN LABDUMP $label====="
  (cd "$dir" && echo "$files" | tar -czf - --no-recursion -T -) | base64 -w 120
  echo "=====END LABDUMP $label====="
}

retrain() {
  case "$job" in
    load@s[0-3] | pv_fc) $PY train.py --preset "$job" ;;
    windnbm) $PY -m wattsentry.sim.wind --no-plots &&
      $PY train.py --model windnbm --epochs 40 --normal-select label_free --seed 0 ;;
    windfailure) $PY train.py --model windfailure --residuals nbm ;;
    *) echo "unknown job $job" >&2; return 2 ;;
  esac
}

case "$job" in  # the original run(s) this job stands for (#141 / #143 / #144)
  load@s[0-3]) own=$((680 + ${job#load@s})) ;;
  pv_fc) own=684 ;;
  windnbm) own="674 675" ;;  # #143: seed 0 or seed 1, either is promote-ready
  windfailure) own=778 ;;
  *) echo "unknown job $job" >&2; exit 2 ;;
esac

echo "== worker $(hostname) work=$WORK job=$job own=$own"
echo "== outputs here: $(ls "$OUT" 2>/dev/null | tr '\n' ' ')"
found_own=0
for id in $TARGETS; do
  if [ -d "$OUT/$id" ]; then
    dump "$id" "$OUT/$id"
    echo "LAB_METRIC fetched_$id=1"
    for o in $own; do [ "$o" = "$id" ] && found_own=1; done
  fi
done

if [ "$found_own" = 1 ]; then
  echo "LAB_METRIC retrained=0"
  exit 0
fi
echo "== $job: original run(s) $own not on this worker; retraining with the original command"
echo "LAB_METRIC retrained=1"
retrain
rc=$?
out="${LAB_OUTPUT_DIR:-.lab-out}"
[ -d "$out" ] && dump "new-$job-r${LAB_RUN_ID:-0}" "$out"
exit $rc
