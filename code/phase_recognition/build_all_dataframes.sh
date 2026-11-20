#!/bin/bash
# ============================================================================
# Build per-camera 1-fps dataframes for the vitrectomy LOO pipeline.
# Run once before run_expAB.sh / run_loo_cv.sh.
#
#   annotations -> annotations/            (override: ANN_ROOT)
#   dataframes  -> data/vitrectomy/dataframes  (override: OUT_DIR)
#                  (the CNN dataset class hardcodes the "dataframes" subdir, so
#                   the pkls must sit under <root>/dataframes/)
#
# Usage:
#   bash build_all_dataframes.sh                       # all 5 cameras
#   bash build_all_dataframes.sh camera01 camera03     # only those
# ============================================================================
set -u -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-$ROOT/venv_tecno/bin/python}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/raw}"
ANN_ROOT="${ANN_ROOT:-$ROOT/annotations}"
OUT_DIR="${OUT_DIR:-$ROOT/data/vitrectomy/dataframes}"

CAMERAS=("$@")
if [ ${#CAMERAS[@]} -eq 0 ]; then
  CAMERAS=(camera01 camera02 camera03 camera04 camera05)
fi

echo "Building dataframes for: ${CAMERAS[*]}"
echo "  dataset : $DATASET_ROOT"
echo "  ann-root: $ANN_ROOT"
echo "  out-dir : $OUT_DIR"
OK=()
FAIL=()
for CAM in "${CAMERAS[@]}"; do
  echo ""
  echo "================  $CAM  ================"
  if "$PY" "$ROOT/vitrectomy/build_dataframe.py" --camera "$CAM" \
        --dataset-root "$DATASET_ROOT" --ann-root "$ANN_ROOT" --out-dir "$OUT_DIR"; then
    OK+=("$CAM")
  else
    FAIL+=("$CAM")
    echo ">>> $CAM: build failed (likely: no JPGs for this camera). Continuing."
  fi
done
echo ""
echo "=============================================================="
echo " build_all_dataframes summary"
[ ${#OK[@]}   -gt 0 ] && echo "   built : ${OK[*]}"
[ ${#FAIL[@]} -gt 0 ] && echo "   failed: ${FAIL[*]}   (these cameras probably don't exist in the raw data)"
echo "=============================================================="
