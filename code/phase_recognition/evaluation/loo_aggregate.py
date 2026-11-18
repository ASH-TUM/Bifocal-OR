"""
Aggregate all LOO folds into one pooled confusion matrix + per-class
report, plus a per-fold accuracy table.

Additionally reports segmental workflow metrics (Lea et al. 2017,
also reported in Czempiel et al. 2020):
  - Edit Score (normalised Levenshtein distance on the per-segment
    label sequence)
  - F1@10, F1@25, F1@50 (segmental F1 at IoU thresholds 0.1, 0.25, 0.5)

Set --ood true to aggregate the OOD predictions (fold<k>_preds_ood.pkl)
instead of the in-domain ones — used for Experiment B reporting.
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# Make project root importable when running as `python evaluation/loo_aggregate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.vitrectomy_constants import CLASS_LABELS


# ──────────────────────────────────────────────────────────────────────
# Segmental metrics (Lea et al. 2017)
# ──────────────────────────────────────────────────────────────────────

def _segments(seq):
    """Return list of (label, start, end_inclusive) for contiguous runs."""
    segs, i = [], 0
    seq = np.asarray(seq)
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1] == seq[i]:
            j += 1
        segs.append((int(seq[i]), i, j))
        i = j + 1
    return segs


def edit_score(pred, gt):
    """Normalised Levenshtein on segment-label sequences. 100 = perfect."""
    P = [s[0] for s in _segments(pred)]
    G = [s[0] for s in _segments(gt)]
    m, n = len(P), len(G)
    if max(m, n) == 0:
        return 100.0
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if P[i - 1] == G[j - 1] else 1
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)
    return 100.0 * (1 - D[m, n] / max(m, n))


def f1_at_k(pred, gt, overlap):
    """Lea et al. 2017 segmental F1@overlap (IoU threshold)."""
    P = _segments(pred)
    G = _segments(gt)
    tp = fp = 0
    matched = [False] * len(G)
    for (lp, ps, pe) in P:
        best_iou, best_j = 0.0, -1
        for j, (lg, gs, ge) in enumerate(G):
            if lg != lp or matched[j]:
                continue
            inter = max(0, min(pe, ge) - max(ps, gs) + 1)
            union = max(pe, ge) - min(ps, gs) + 1
            iou = inter / union if union else 0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= overlap and best_j >= 0:
            tp += 1
            matched[best_j] = True
        else:
            fp += 1
    fn = matched.count(False)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return 200 * prec * rec / (prec + rec) if prec + rec else 0


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--ood", action="store_true",
                    help="Aggregate fold<k>_preds_ood.pkl instead of "
                         "fold<k>_preds.pkl (Experiment B reporting).")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    suffix = "_ood" if args.ood else ""
    fold_files = sorted(rd.glob(f"fold*_preds{suffix}.pkl"))
    if not fold_files:
        raise SystemExit(f"No fold*_preds{suffix}.pkl found in {rd}")

    all_true, all_pred = [], []
    per_fold = []
    seg_metrics = []
    for f in fold_files:
        d = pickle.loads(f.read_bytes())
        yt, yp = np.asarray(d["y_true"]), np.asarray(d["y_pred"])
        all_true.append(yt)
        all_pred.append(yp)
        acc = float((yt == yp).mean())
        ed = edit_score(yp, yt)
        f10 = f1_at_k(yp, yt, 0.10)
        f25 = f1_at_k(yp, yt, 0.25)
        f50 = f1_at_k(yp, yt, 0.50)
        per_fold.append((d["test_fold"], acc, len(yt)))
        seg_metrics.append((d["test_fold"], ed, f10, f25, f50))

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    labels = list(range(len(CLASS_LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    lines = []
    lines.append("=" * 78)
    title = "LEAVE-ONE-OUT CV"
    if args.ood:
        title += " — OUT-OF-DOMAIN (Experiment B: Run 9 zero-shot)"
    else:
        title += " — vitrectomy phase recognition"
    lines.append(title)
    lines.append("=" * 78)
    lines.append("\nPer-fold accuracy:")
    for fold, acc, n in per_fold:
        lines.append(f"  fold {fold}  acc={acc:.4f}   ({n} frames)")
    accs = np.array([a for _, a, _ in per_fold])
    lines.append(f"\n  mean fold accuracy : {accs.mean():.4f}  "
                 f"(sample std {accs.std(ddof=1):.4f})")
    pooled_acc = float((y_true == y_pred).mean())
    lines.append(f"  pooled frame accuracy: {pooled_acc:.4f}  "
                 f"({len(y_true)} frames over all folds)")

    # ── Segmental metrics ────────────────────────────────────────────
    lines.append("\nSegmental metrics (Lea et al. 2017):")
    lines.append(f"  {'fold':>5} {'Edit':>7} {'F1@10':>7} {'F1@25':>7} {'F1@50':>7}")
    arr = np.array([(e, a, b, c) for _, e, a, b, c in seg_metrics])
    for fold, e, a, b, c in seg_metrics:
        lines.append(f"  {fold:>5} {e:>7.2f} {a:>7.2f} {b:>7.2f} {c:>7.2f}")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(mean)
    lines.append(f"  {'mean':>5} {mean[0]:>7.2f} {mean[1]:>7.2f} {mean[2]:>7.2f} {mean[3]:>7.2f}")
    lines.append(f"  {'std':>5} {std[0]:>7.2f} {std[1]:>7.2f} {std[2]:>7.2f} {std[3]:>7.2f}")

    # ── Confusion + classification report ────────────────────────────
    lines.append("\nPooled confusion matrix (rows = true, cols = predicted):")
    lines.append("             " + "".join(f"{l[:10]:>12}" for l in CLASS_LABELS) + "   total")
    for i, lbl in enumerate(CLASS_LABELS):
        row = f"{lbl:>12} " + "".join(f"{v:>12d}" for v in cm[i]) + f"{cm[i].sum():>8d}"
        lines.append(row)

    lines.append("\nPer-class precision / recall / f1 (pooled):")
    lines.append(classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_LABELS,
        digits=3, zero_division=0,
    ))

    report = "\n".join(lines)
    print(report)
    out_txt = rd / f"loo_summary{suffix}.txt"
    out_txt.write_text(report)

    # ── PNG ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False, xticks_rotation=30)
    suptitle = "OOD (zero-shot Run 9)" if args.ood else "LOO-CV pooled"
    ax.set_title(f"{suptitle} confusion matrix\npooled acc = {pooled_acc:.3f}  "
                 f"| mean fold acc = {accs.mean():.3f}")
    plt.tight_layout()
    out_png = rd / f"loo_confusion_matrix{suffix}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
