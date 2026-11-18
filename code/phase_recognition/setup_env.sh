#!/bin/bash
# ============================================================================
# One-time environment setup for the vitrectomy LOO-CV pipeline (local or server).
# Run once after cloning the repo:   bash setup_env.sh
#
# Override the base interpreter with:  PYBIN=python3.11 bash setup_env.sh
# ============================================================================
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/venv_tecno"
PYBIN="${PYBIN:-python3}"

if [ -d "$VENV" ] && [ -x "$VENV/bin/python" ]; then
  echo "venv already exists at $VENV — reusing it (delete the folder to rebuild)."
else
  echo "Creating venv at $VENV using $PYBIN ..."
  "$PYBIN" -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"

echo ""
"$VENV/bin/python" - <<'EOF'
import torch, pytorch_lightning as pl
print(f"torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"pytorch-lightning {pl.__version__}")
EOF

echo ""
echo "Environment ready. Sanity-check one fold quickly with:"
echo "  venv_tecno/bin/python train.py -c modules/cnn/config/config_feature_extract_vitrectomy.yml --test_fold 0 --fast_dev_run True"
echo "Then launch the full LOO-CV:"
echo "  bash run_loo_cv.sh"
