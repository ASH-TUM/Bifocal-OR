"""
Evaluate one LOO fold: load the Stage 2 (MS-TCN) checkpoint and run
inference on the held-out test video. Optionally also evaluate the same
model on an out-of-domain video (Experiment B: Run 9 = video_8) without
any retraining.

Output pickles always contain:
    {"test_fold": int, "y_true": np.ndarray, "y_pred": np.ndarray, "y_soft": np.ndarray}

The OOD output records the in-domain test_fold the model was trained
for and a separate `ood_video` field so downstream analysis knows
where the prediction came from.
"""
import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

# Make project root importable when running as `python evaluation/loo_evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from models.mstcn import MultiStageModel


def evaluate(model, features_dir: Path, video_idx: int):
    """Run the model on `video_<video_idx>_1.0fps.pkl` features.

    Returns (y_true, y_pred, y_soft) for labeled frames only.
    """
    pkl = features_dir / f"video_{video_idx}_1.0fps.pkl"
    stems, _, ys = pd.read_pickle(pkl)
    stems = torch.from_numpy(np.asarray(stems, dtype=np.float32))
    ys = np.asarray(ys)

    x = stems.unsqueeze(0).transpose(1, 2)        # [1, 2048, N]
    with torch.no_grad():
        out = torch.softmax(model(x), dim=2)      # [stages, 1, C, N]
    soft = out[-1, 0].numpy()                     # [C, N]
    preds = soft.argmax(axis=0)                   # [N]

    mask = ys >= 0
    return ys[mask], preds[mask], soft[:, mask].T  # [n, 5]


def dump(path: Path, payload: dict):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_fold", type=int, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--features", type=str, required=True,
                    help="Path to the 1.0fps feature folder for the in-domain "
                         "test video.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output pickle for the in-domain test fold.")
    ap.add_argument("--ood_video", type=int, default=None,
                    help="Optional out-of-domain video index to evaluate the "
                         "same model on (Exp B; usually 8 = Run 9).")
    ap.add_argument("--ood_features", type=str, default=None,
                    help="Feature folder for the OOD video. If omitted, "
                         "defaults to --features.")
    ap.add_argument("--ood_out", type=str, default=None,
                    help="Output pickle for the OOD evaluation. Required if "
                         "--ood_video is set.")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hp = SimpleNamespace(**ckpt["hyper_parameters"])
    model = MultiStageModel(hp)
    state = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    model.load_state_dict(state)
    model.eval()

    # ── In-domain evaluation ──────────────────────────────────────────
    y_true, y_pred, y_soft = evaluate(model, Path(args.features), args.test_fold)
    acc = float((y_true == y_pred).mean())
    dump(Path(args.out), {
        "test_fold": args.test_fold,
        "y_true": y_true, "y_pred": y_pred, "y_soft": y_soft,
    })
    print(f"In-domain fold {args.test_fold}: acc={acc:.4f} "
          f"({len(y_true)} frames) -> {args.out}")

    # ── OOD evaluation (optional) ─────────────────────────────────────
    if args.ood_video is not None:
        if args.ood_out is None:
            raise SystemExit("--ood_out is required when --ood_video is set")
        ood_features = Path(args.ood_features or args.features)
        y_true_o, y_pred_o, y_soft_o = evaluate(model, ood_features, args.ood_video)
        acc_o = float((y_true_o == y_pred_o).mean())
        dump(Path(args.ood_out), {
            "test_fold": args.test_fold,        # the model was trained for this fold
            "ood_video": args.ood_video,        # but evaluated on this video
            "y_true": y_true_o, "y_pred": y_pred_o, "y_soft": y_soft_o,
        })
        print(f"OOD video_{args.ood_video} (model from fold {args.test_fold}): "
              f"acc={acc_o:.4f} ({len(y_true_o)} frames) -> {args.ood_out}")


if __name__ == "__main__":
    main()
