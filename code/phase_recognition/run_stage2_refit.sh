#!/bin/bash
# ============================================================================
# Stage-2-only retraining on existing Stage-1 features.
#
# Final protocol: MS-TCN is trained with the Stage-2 config as shipped
# (seed 5678 + fold, early-stopping patience 30). Stage 1 is NOT retrained:
# each fold re-uses the 2048-d feature export of the original run_expAB.sh
# run, located via the "Output path:" line of that run's fold<k>_stage1.log.
#
# Experiment A/B setup as in run_expAB.sh: pool = Runs 1-8 (video_0..7),
# 1 validation video, Run 9 (video_8) = zero-shot OOD test for every fold.
#
# Usage:
#   bash run_stage2_refit.sh                        # all cameras x 8 folds
#   bash run_stage2_refit.sh camera03               # camera subset
#   bash run_stage2_refit.sh camera03 -- 0 3        # camera + fold subset
#
# Env:
#   OUTPUTS   base dir of the original run (default: ./outputs); must contain
#             expAB_loo_cv_<camera>/fold<k>_stage1.log
#   REFIT_DIR where predictions go (default: $OUTPUTS/refit_all/<camera>)
#   PYTHON    interpreter (default: ./venv_tecno/bin/python)
#   SEED      base Stage-2 seed passed as --seed (default: the config's seed,
#             5678). Overrides the config on the command line, so several
#             seeds can run side by side without editing the config file.
#
# Output (same format as run_expAB.sh):
#   $REFIT_DIR/<camera>/fold<k>_preds.pkl, fold<k>_preds_ood.pkl, fold<k>_stage2.log
# Resumable: folds with an existing preds pkl are skipped.
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/venv_tecno/bin/python}"
OUTPUTS="${OUTPUTS:-$ROOT/outputs}"
REFIT_DIR="${REFIT_DIR:-$OUTPUTS/refit_all}"
CFG2="$ROOT/modules/mstcn/config/config_tcn_vitrectomy.yml"
PICKLE_FOLDER="vitrectomy_pickle_export"
POOL="0,1,2,3,4,5,6,7"
SEED_ARG=()
[ -n "${SEED:-}" ] && SEED_ARG=(--seed "$SEED")
VAL_SIZE=1
OOD_VIDEO=8

CAMERAS=(); FOLDS=(); SAW_SEP=0
for a in "$@"; do
  if [ "$a" = "--" ]; then SAW_SEP=1; continue; fi
  if [ "$SAW_SEP" -eq 0 ]; then CAMERAS+=("$a"); else FOLDS+=("$a"); fi
done
[ ${#CAMERAS[@]} -eq 0 ] && CAMERAS=(camera01 camera02 camera03 camera04 camera05)
[ ${#FOLDS[@]} -eq 0 ] && FOLDS=(0 1 2 3 4 5 6 7)

if [ ! -x "$PY" ]; then
  echo "ERROR: Python not found at $PY (run setup_env.sh or set PYTHON)."; exit 1
fi
cd "$ROOT"

echo "=============================================================="
echo " Stage-2 refit | cameras: ${CAMERAS[*]} | folds: ${FOLDS[*]}"
echo " features from : $OUTPUTS/expAB_loo_cv_<camera>/fold<k>_stage1.log"
echo " results       : $REFIT_DIR/<camera>"
echo " stage-2 seed  : ${SEED:-from config}"
echo " started       : $(date -Is)"
echo "=============================================================="

FAILED=()
for CAM in "${CAMERAS[@]}"; do
  RES="$REFIT_DIR/$CAM"
  mkdir -p "$RES"
  for k in "${FOLDS[@]}"; do
    PREDS="$RES/fold${k}_preds.pkl"
    if [ -f "$PREDS" ]; then echo "[$CAM fold $k] already done, skipping"; continue; fi

    # Stage-1 feature export of the original run for this camera/fold
    LOG1="$OUTPUTS/expAB_loo_cv_$CAM/fold${k}_stage1.log"
    OUT1=$(grep -oP '^Output path:\s+\K\S+' "$LOG1" 2>/dev/null | tail -1)
    FEAT="$OUT1/$PICKLE_FOLDER"
    if [ -z "$OUT1" ] || [ ! -d "$FEAT/1.0fps" ]; then
      echo "[$CAM fold $k] ERROR: Stage-1 features not found (log: $LOG1)"
      FAILED+=("$CAM:$k:features"); continue
    fi

    LOG2="$RES/fold${k}_stage2.log"
    echo "[$CAM fold $k] $(date -Is)  Stage 2 on $FEAT"
    "$PY" train.py -c "$CFG2" --test_fold "$k" --data_root "$FEAT" --camera "$CAM" \
          --val_size "$VAL_SIZE" --pool "$POOL" "${SEED_ARG[@]}" 2>&1 | tee "$LOG2"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then FAILED+=("$CAM:$k:stage2"); continue; fi

    OUT2=$(grep -oP '^Output path:\s+\K\S+' "$LOG2" | tail -1)
    CKPT=$(ls -t "$OUT2"/checkpoints/*.ckpt 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then FAILED+=("$CAM:$k:checkpoint"); continue; fi

    "$PY" evaluation/loo_evaluate.py --test_fold "$k" --ckpt "$CKPT" \
          --features "$FEAT/1.0fps" --out "$PREDS" \
          --ood_video "$OOD_VIDEO" --ood_features "$FEAT/1.0fps" \
          --ood_out "$RES/fold${k}_preds_ood.pkl"
    if [ "$?" -ne 0 ] || [ ! -f "$PREDS" ]; then FAILED+=("$CAM:$k:evaluate"); continue; fi
  done
  "$PY" evaluation/loo_aggregate.py --results_dir "$RES" || true
  "$PY" evaluation/loo_aggregate.py --results_dir "$RES" --ood || true
done

echo "=============================================================="
echo " finished: $(date -Is)"
if [ ${#FAILED[@]} -gt 0 ]; then echo " FAILED: ${FAILED[*]}"; fi
echo "=============================================================="
[ ${#FAILED[@]} -eq 0 ]
