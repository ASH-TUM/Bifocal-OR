#!/bin/bash
# ============================================================================
# Leave-One-Out cross-validation for the vitrectomy TeCNO pipeline.
#
# Usage:
#   bash run_loo_cv.sh            # all 9 folds
#   bash run_loo_cv.sh 0 1 2      # only folds 0, 1, 2
#
# Reproducible: each fold uses a fixed per-fold seed (config `seed` + fold).
# Resumable:    a fold whose fold<k>_preds.pkl already exists is skipped.
# Override the interpreter with:  PYTHON=/path/to/python bash run_loo_cv.sh
# ============================================================================
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/venv_tecno/bin/python}"
CFG1="$ROOT/modules/cnn/config/config_feature_extract_vitrectomy.yml"
CFG2="$ROOT/modules/mstcn/config/config_tcn_vitrectomy.yml"
# Camera and results dir can be overridden by the multi-camera wrapper
# (run_expAB.sh). Defaults preserve a single-camera baseline.
CAMERA="${CAMERA:-camera01}"
RESULTS="${RESULTS:-${OUTPUTS:-$ROOT/outputs}/loo_cv}"
PICKLE_FOLDER="vitrectomy_pickle_export"   # must match pickle_export_folder in CFG1

# Experiment A/B knobs (default empty = 9-fold leave-one-out):
#   POOL       comma- or space-separated list of video indices the LOO
#              rotation operates over. Empty = all available videos.
#              Exp A/B: "0,1,2,3,4,5,6,7"  (Runs 1-8, Run 9 reserved OOD)
#   VAL_SIZE   number of cyclic val videos per fold. default 2; Exp A/B uses 1
#   OOD_VIDEO  if set (e.g. 8 = Run 9), each fold ALSO evaluates the trained
#              model on this video and dumps fold<k>_preds_ood.pkl. The pool
#              should exclude this video index — if not, the script errors.
POOL="${POOL:-}"
VAL_SIZE="${VAL_SIZE:-2}"
OOD_VIDEO="${OOD_VIDEO:-}"

mkdir -p "$RESULTS"
cd "$ROOT"

# Safety: OOD_VIDEO must not be inside POOL
if [ -n "$OOD_VIDEO" ] && [ -n "$POOL" ]; then
  if echo "$POOL" | tr ',' ' ' | tr -s ' ' '\n' | grep -qx "$OOD_VIDEO"; then
    echo "ERROR: OOD_VIDEO=$OOD_VIDEO is also in POOL=$POOL — that's a leak."
    exit 1
  fi
fi

if [ ! -x "$PY" ]; then
  echo "ERROR: Python interpreter not found/executable at: $PY"
  echo "Run  'bash setup_env.sh'  first, or set PYTHON=/path/to/python."
  exit 1
fi

FOLDS=("$@")
if [ ${#FOLDS[@]} -eq 0 ]; then
  FOLDS=(0 1 2 3 4 5 6 7 8)
fi

echo "=============================================================="
echo " LOO-CV  |  camera: $CAMERA  |  folds: ${FOLDS[*]}"
echo " pool     : ${POOL:-(all)}    val_size: $VAL_SIZE   OOD video: ${OOD_VIDEO:-(none)}"
echo " python   : $PY"
echo " results  : $RESULTS"
echo " started  : $(date -Is)"
echo "=============================================================="

FAILED=()
SKIPPED=()

for k in "${FOLDS[@]}"; do
  PREDS="$RESULTS/fold${k}_preds.pkl"
  if [ -f "$PREDS" ]; then
    echo ""
    echo "########  FOLD $k  ->  already done, skipping ($PREDS)  ########"
    SKIPPED+=("$k")
    continue
  fi

  echo ""
  echo "################  FOLD $k  (test = video_$k)  ################"
  LOG1="$RESULTS/fold${k}_stage1.log"
  LOG2="$RESULTS/fold${k}_stage2.log"

  # ---- Stage 1: CNN training + feature extraction ----
  echo "[fold $k] $(date -Is)  Stage 1: CNN feature extraction (camera=$CAMERA) ..."
  S1_EXTRA=(--test_fold "$k" --camera "$CAMERA" --val_size "$VAL_SIZE")
  [ -n "$POOL" ] && S1_EXTRA+=(--pool "$POOL")
  "$PY" train.py -c "$CFG1" "${S1_EXTRA[@]}" 2>&1 | tee "$LOG1"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[fold $k] ERROR: Stage 1 crashed (see $LOG1)"
    FAILED+=("$k:stage1"); continue
  fi
  # Derive the feature folder from the (single, reliable) "Output path:" line.
  OUT1=$(grep -oP '^Output path:\s+\K\S+' "$LOG1" | tail -1)
  FEAT="$OUT1/$PICKLE_FOLDER"
  if [ -z "$OUT1" ] || [ ! -d "$FEAT/1.0fps" ]; then
    echo "[fold $k] ERROR: feature export folder not found ($FEAT/1.0fps)"
    FAILED+=("$k:features"); continue
  fi
  echo "[fold $k] features -> $FEAT"

  # ---- Stage 2: MS-TCN training ----
  echo "[fold $k] $(date -Is)  Stage 2: MS-TCN training (camera=$CAMERA) ..."
  S2_EXTRA=(--test_fold "$k" --data_root "$FEAT" --camera "$CAMERA" --val_size "$VAL_SIZE")
  [ -n "$POOL" ] && S2_EXTRA+=(--pool "$POOL")
  "$PY" train.py -c "$CFG2" "${S2_EXTRA[@]}" 2>&1 | tee "$LOG2"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[fold $k] ERROR: Stage 2 crashed (see $LOG2)"
    FAILED+=("$k:stage2"); continue
  fi
  OUT2=$(grep -oP '^Output path:\s+\K\S+' "$LOG2" | tail -1)
  CKPT=$(ls -t "$OUT2"/checkpoints/*.ckpt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "[fold $k] ERROR: no Stage 2 checkpoint in $OUT2/checkpoints/"
    FAILED+=("$k:checkpoint"); continue
  fi
  echo "[fold $k] checkpoint -> $CKPT"

  # ---- Evaluate the held-out test video (+ optional OOD) ----
  EVAL_EXTRA=(--test_fold "$k" --ckpt "$CKPT" --features "$FEAT/1.0fps" --out "$PREDS")
  if [ -n "$OOD_VIDEO" ]; then
    OOD_PREDS="$RESULTS/fold${k}_preds_ood.pkl"
    EVAL_EXTRA+=(--ood_video "$OOD_VIDEO" --ood_features "$FEAT/1.0fps" --ood_out "$OOD_PREDS")
  fi
  "$PY" evaluation/loo_evaluate.py "${EVAL_EXTRA[@]}"
  if [ "$?" -ne 0 ] || [ ! -f "$PREDS" ]; then
    echo "[fold $k] ERROR: evaluation failed"
    FAILED+=("$k:evaluate"); continue
  fi
  echo "[fold $k] DONE -> $PREDS"
  [ -n "$OOD_VIDEO" ] && echo "[fold $k] OOD  -> $RESULTS/fold${k}_preds_ood.pkl"
done

# ---- Aggregate whatever folds are complete ----
echo ""
echo "################  AGGREGATING COMPLETED FOLDS  ################"
"$PY" evaluation/loo_aggregate.py --results_dir "$RESULTS" || true

echo ""
echo "=============================================================="
echo " LOO-CV finished: $(date -Is)"
[ ${#SKIPPED[@]} -gt 0 ] && echo " skipped (already done): ${SKIPPED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo " FAILED: ${FAILED[*]}"
  echo " -> fix the cause, then re-run  bash run_loo_cv.sh ${FAILED[*]%%:*}"
else
  echo " all requested folds completed."
fi
echo " results : $RESULTS/loo_summary.txt"
echo "           $RESULTS/loo_confusion_matrix.png"
echo "=============================================================="
[ ${#FAILED[@]} -eq 0 ]
