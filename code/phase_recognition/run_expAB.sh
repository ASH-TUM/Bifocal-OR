#!/bin/bash
# ============================================================================
# Experiment A + B in one run: 8-fold in-domain LOO-CV over Runs 1-8, plus
# zero-shot OOD evaluation on Run 9 (pig eye), for one or more cameras.
#
# Experiment A (in-domain LOOCV):
#   - Pool = Runs 1-8 (video_0 .. video_7), Run 9 (video_8) is held out.
#   - 8 folds per camera. Each fold: 6 train + 1 val + 1 in-domain test.
# Experiment B (zero-shot OOD on Run 9):
#   - Each of the 8 fold-models is also evaluated on Run 9 features without
#     retraining -> 8 OOD predictions per camera.
#
# Pre-requisites (run once on the server):
#   bash setup_env.sh
#   bash build_all_dataframes.sh
#
# Usage:
#   bash run_expAB.sh                          # all 5 cameras x 8 folds
#   bash run_expAB.sh camera01 camera03        # camera subset
#   bash run_expAB.sh -- 0 3 7                 # only those folds
#   bash run_expAB.sh camera03 -- 0 3          # both
#
# Output:
#   outputs/expAB_loo_cv_<camera>/
#       fold<k>_preds.pkl       in-domain (held-out plastic run)
#       fold<k>_preds_ood.pkl   OOD (Run 9 = video_8)
#       loo_summary.txt         in-domain aggregate (auto)
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/venv_tecno/bin/python}"
OUTPUTS="${OUTPUTS:-$ROOT/outputs}"

# Experiment A: pool = Runs 1-8 (video indices 0..7).
# Experiment B: Run 9 = video_8 is the fixed OOD test.
EXP_POOL="0,1,2,3,4,5,6,7"
EXP_VAL_SIZE=1
EXP_OOD_VIDEO=8
EXP_FOLDS_DEFAULT=(0 1 2 3 4 5 6 7)

# Parse positional args: [cameras...] -- [folds...]
CAMERAS=()
FOLDS=()
SAW_SEP=0
for a in "$@"; do
  if [ "$a" = "--" ]; then SAW_SEP=1; continue; fi
  if [ "$SAW_SEP" -eq 0 ]; then CAMERAS+=("$a"); else FOLDS+=("$a"); fi
done
if [ ${#CAMERAS[@]} -eq 0 ]; then
  CAMERAS=(camera01 camera02 camera03 camera04 camera05)
fi
if [ ${#FOLDS[@]} -eq 0 ]; then
  FOLDS=("${EXP_FOLDS_DEFAULT[@]}")
fi

# Preflight: dataframes exist? Derive the dir from the CNN config.
CFG1="$ROOT/modules/cnn/config/config_feature_extract_vitrectomy.yml"
DATA_ROOT=$(grep -oP '^data_root:\s*\K\S+' "$CFG1" | head -1)
case "$DATA_ROOT" in /*) ;; *) DATA_ROOT="$ROOT/$DATA_ROOT" ;; esac
DF_DIR="${DF_DIR:-$DATA_ROOT/dataframes}"

AVAILABLE=()
MISSING=()
for CAM in "${CAMERAS[@]}"; do
  if [ -f "$DF_DIR/vitrectomy_1fps_$CAM.pkl" ]; then
    AVAILABLE+=("$CAM")
  else
    MISSING+=("$CAM")
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo ""
  echo "WARNING: skipping cameras with no dataframe in $DF_DIR:"
  for CAM in "${MISSING[@]}"; do
    echo "   - $CAM   (expected $DF_DIR/vitrectomy_1fps_$CAM.pkl)"
  done
  echo "Build missing dataframes with:  bash build_all_dataframes.sh ${MISSING[*]}"
  echo ""
fi
if [ ${#AVAILABLE[@]} -eq 0 ]; then
  echo "ERROR: no dataframes found in $DF_DIR. Run  bash build_all_dataframes.sh  first."
  exit 1
fi
CAMERAS=("${AVAILABLE[@]}")

echo "=============================================================="
echo " Experiment A + B"
echo "   cameras    : ${CAMERAS[*]}"
echo "   folds (A)  : ${FOLDS[*]}     (pool = $EXP_POOL, val_size = $EXP_VAL_SIZE)"
echo "   OOD (B)    : video_$EXP_OOD_VIDEO (= Run 9, pig eye), every fold"
echo "   started    : $(date -Is)"
echo "=============================================================="

CAM_FAILED=()
for CAM in "${CAMERAS[@]}"; do
  RES="$OUTPUTS/expAB_loo_cv_$CAM"
  echo ""
  echo "##############################################################"
  echo "##  CAMERA $CAM  ->  $RES"
  echo "##############################################################"

  POOL="$EXP_POOL" \
  VAL_SIZE="$EXP_VAL_SIZE" \
  OOD_VIDEO="$EXP_OOD_VIDEO" \
  CAMERA="$CAM" \
  RESULTS="$RES" \
    bash "$ROOT/run_loo_cv.sh" "${FOLDS[@]}"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[$CAM] ended with non-zero exit ($rc) — some folds may have failed."
    CAM_FAILED+=("$CAM")
  fi
done

# Cross-camera + OOD comparison
echo ""
echo "##############################################################"
echo "##  CROSS-CAMERA COMPARISON"
echo "##############################################################"
"$PY" "$ROOT/evaluation/loo_voting.py" compare-cameras --root "$OUTPUTS" \
      --pattern "expAB_loo_cv_camera*" \
      --out "$OUTPUTS/expAB_summary" 2>/dev/null || true

echo ""
echo "=============================================================="
echo " Exp A+B finished: $(date -Is)"
if [ ${#CAM_FAILED[@]} -gt 0 ]; then
  echo "  cameras with at least one failed fold: ${CAM_FAILED[*]}"
fi
echo "  In-domain results:  outputs/expAB_loo_cv_<camera>/loo_summary.txt"
echo "  OOD predictions  :  outputs/expAB_loo_cv_<camera>/fold<k>_preds_ood.pkl"
echo "=============================================================="
