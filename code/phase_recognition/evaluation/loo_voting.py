"""
Consolidated camera-ensembling experiments for the vitrectomy LOO
predictions. Every subcommand only reads the fold<k>_preds.pkl files
loo_evaluate.py already wrote ({y_true, y_pred, y_soft}) — never loads
a model, so everything here is compute-free (seconds, no GPU).

Subcommands (== Experiments 2-5 and 7 of the analysis pipeline):

  compare-cameras   Exp 2 — per-camera accuracy, no voting.
  static            Exp 3 — hard vs soft vote, brute-force camera subsets.
  dynamic           Exp 4 — entropy-ranked top-k vote, windows 0/N/inf.
  pf-vs-pv          Exp 5 — per-frame (Δ=0) vs per-video (Δ=∞) diagnostics.
  dropout           Exp 7 — sensor-dropout robustness (drop-2 scenarios).

Segmental metrics (Exp 6: Edit Score, F1@10/25/50) live in loo_aggregate.py
and are imported here rather than duplicated.

Usage:
  python evaluation/loo_voting.py compare-cameras --root outputs
  python evaluation/loo_voting.py static --root outputs
  python evaluation/loo_voting.py dynamic --pattern 'loo_cv_camera*'
  python evaluation/loo_voting.py pf-vs-pv --pattern 'loo_cv_camera*' --ood
  python evaluation/loo_voting.py dropout --pattern 'loo_cv_camera*'
"""
import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Make project root importable when running as `python evaluation/loo_voting.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loo_aggregate import edit_score, f1_at_k
from utils.vitrectomy_constants import CLASS_LABELS as CLASSES, DEFAULT_CAMERAS

EPS = 1e-12
RUN_LABELS = {
    0: "Run 1", 1: "Run 2", 2: "Run 3",
    3: "Run 4 (visitor chaos)", 4: "Run 5", 5: "Run 6", 6: "Run 7",
    7: "Run 8 (trainee)", 8: "Run 9 (pig eye / OOD)",
}


# ══════════════════════════════════════════════════════════════════════
# Shared IO
# ══════════════════════════════════════════════════════════════════════

def load_fold_exact(root: Path, cam: str, k: int):
    """<root>/loo_cv_<cam>/fold<k>_preds.pkl — used by `static`."""
    p = root / f"loo_cv_{cam}" / f"fold{k}_preds.pkl"
    if not p.exists():
        return None
    return pickle.loads(p.read_bytes())


def load_fold_by_pattern(root: Path, pattern: str, cam: str, k: int, ood: bool):
    """Glob-resolve <root>/<pattern matching cam>/fold<k>_preds[_ood].pkl.
    Used by `dynamic`, `pf-vs-pv`, `dropout` (Exp A/B result dirs)."""
    suffix = "_ood" if ood else ""
    base_pattern = pattern.rstrip("*").rstrip("_")
    candidates = list(root.glob(f"{base_pattern}_{cam}"))
    if not candidates:
        candidates = [d for d in root.glob(pattern) if d.name.endswith(cam)]
    if not candidates:
        return None
    p = candidates[0] / f"fold{k}_preds{suffix}.pkl"
    if not p.exists():
        return None
    return pickle.loads(p.read_bytes())


# ══════════════════════════════════════════════════════════════════════
# Shared metrics / voting primitives
# ══════════════════════════════════════════════════════════════════════

def pooled_metrics(yt, yp):
    yt = np.asarray(yt); yp = np.asarray(yp)
    return {
        "acc": float((yt == yp).mean()),
        "macro_f1": float(f1_score(yt, yp, labels=list(range(len(CLASSES))),
                                   average="macro", zero_division=0)),
        "n": int(len(yt)),
    }


def softmax_entropy(p):
    """p: [N, C] softmax → returns H per frame, shape [N]."""
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def smooth(signal, window):
    """Symmetric running mean over [t-window, t+window]. window<=0 → identity."""
    if window <= 0:
        return signal
    N = len(signal)
    out = np.zeros(N)
    cum = np.concatenate([[0.0], np.cumsum(signal)])
    for t in range(N):
        a = max(0, t - window)
        b = min(N, t + window + 1)
        out[t] = (cum[b] - cum[a]) / (b - a)
    return out


def dynamic_top_k_vote(per_cam_soft, window, k=3):
    """Confidence-weighted top-k soft vote, ranked by softmax entropy.

    per_cam_soft: dict camera_name -> [N, C] softmax
    window: 0 = strict per-frame, N = smoothed per-frame (±N frames),
            float('inf') = per-video (fixed top-k for the whole video)
    k: how many cameras vote per frame (default 3)

    Returns: (y_pred [N], selection [N, n_cams] bool, entropies [n_cams, N]).
    window=0 reproduces the old `per_frame_vote`, window=inf the old
    `per_video_vote` — both are this same function, so the diagnostics in
    `pf-vs-pv` and the ablation in `dynamic` use byte-identical logic.
    """
    cams = list(per_cam_soft.keys())
    softs = np.stack([per_cam_soft[c] for c in cams], axis=0)   # [C_cams, N, K]
    N = softs.shape[1]
    n_cams = len(cams)

    entropies = np.stack([softmax_entropy(softs[i]) for i in range(n_cams)], axis=0)  # [C_cams, N]
    if window == float("inf"):
        ranks = entropies.mean(axis=1)
        top_idx = np.argsort(ranks)[:k]
        selection = np.zeros((N, n_cams), dtype=bool)
        selection[:, top_idx] = True
        weights = 1.0 / np.clip(entropies.mean(axis=1, keepdims=True), EPS, None)
        weights = np.broadcast_to(weights, (n_cams, N))
    else:
        smoothed = np.stack([smooth(entropies[i], int(window)) for i in range(n_cams)], axis=0)
        order = np.argsort(smoothed, axis=0)        # [C_cams, N], ascending
        selection = np.zeros((N, n_cams), dtype=bool)
        for kk in range(min(k, n_cams)):
            for t in range(N):
                selection[t, order[kk, t]] = True
        weights = 1.0 / np.clip(smoothed, EPS, None)  # [C_cams, N]

    avg = np.zeros((N, softs.shape[2]))
    wsum = np.zeros(N)
    for ci in range(n_cams):
        m = selection[:, ci]
        w = weights[ci] * m
        avg += w[:, None] * softs[ci]
        wsum += w
    avg /= np.clip(wsum[:, None], EPS, None)
    return avg.argmax(axis=1), selection, entropies


def hard_vote(pred_list, weights=None):
    """Majority vote with optional weights."""
    arr = np.stack(pred_list, axis=0)               # [C_cams, N]
    N = arr.shape[1]
    out = np.zeros(N, dtype=int)
    for i in range(N):
        score = {}
        for j in range(arr.shape[0]):
            c = int(arr[j, i])
            w = 1.0 if weights is None else weights[j]
            score[c] = score.get(c, 0.0) + w
        out[i] = max(score, key=score.get)
    return out


def soft_vote(soft_list, weights=None):
    """soft_list: list of [N, 5] softmax arrays. Returns [N] argmax of mean."""
    stack = np.stack(soft_list, axis=0)             # [C_cams, N, 5]
    if weights is not None:
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1, 1)
        avg = (stack * w).sum(axis=0) / w.sum()
    else:
        avg = stack.mean(axis=0)
    return avg.argmax(axis=1)


def confidence_soft_vote(per_cam_soft, weights=None):
    """Dict-keyed plain weighted average (no top-k re-selection) — used by
    `dropout`, where the camera subset is already fixed by the drop."""
    cams = list(per_cam_soft.keys())
    softs = np.stack([per_cam_soft[c] for c in cams], axis=0)
    if weights is None:
        avg = softs.mean(axis=0)
    else:
        w = np.asarray([weights[c] for c in cams]).reshape(-1, 1, 1)
        avg = (softs * w).sum(axis=0) / w.sum()
    return avg.argmax(axis=1)


def boundary_mask(y_true, window=3):
    """True for frames within +/-window of a phase transition."""
    y = np.asarray(y_true)
    diff = np.zeros_like(y, dtype=bool)
    diff[1:] = y[1:] != y[:-1]
    mask = np.zeros_like(y, dtype=bool)
    for idx in np.where(diff)[0]:
        a = max(0, idx - window)
        b = min(len(y), idx + window + 1)
        mask[a:b] = True
    return mask


def selection_switch_rate(sel):
    """Fraction of frame pairs (t, t+1) where the selected-set changes."""
    if len(sel) < 2:
        return 0.0
    return float((sel[1:] != sel[:-1]).any(axis=1).mean())


# ══════════════════════════════════════════════════════════════════════
# Exp 2 — compare-cameras (no voting)
# ══════════════════════════════════════════════════════════════════════

def collect_camera(rd: Path):
    fold_data = []
    for k in range(9):
        p = rd / f"fold{k}_preds.pkl"
        if not p.exists():
            continue
        d = pickle.loads(p.read_bytes())
        fold_data.append((int(d["test_fold"]), d["y_true"], d["y_pred"]))
    if not fold_data:
        return None
    accs_by_fold = {k: float((yt == yp).mean()) for k, yt, yp in fold_data}
    per_fold_acc = [accs_by_fold.get(k, float("nan")) for k in range(9)]
    yts = np.concatenate([yt for _, yt, _ in fold_data])
    yps = np.concatenate([yp for _, _, yp in fold_data])
    accs_arr = np.array([accs_by_fold[k] for k in sorted(accs_by_fold)])
    return {
        "name": rd.name,
        "camera": rd.name.removeprefix("loo_cv_"),
        "n_folds": len(fold_data),
        "per_fold_acc": per_fold_acc,
        "mean_acc": float(accs_arr.mean()),
        "std_acc_pop": float(accs_arr.std(ddof=0)),
        "std_acc_sample": float(accs_arr.std(ddof=1)) if len(accs_arr) > 1 else float("nan"),
        "pooled_acc": float((yts == yps).mean()),
        "macro_f1": float(f1_score(yts, yps, labels=list(range(len(CLASSES))),
                                   average="macro", zero_division=0)),
        "n_frames": int(len(yts)),
    }


def cmd_compare_cameras(args):
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)

    dirs = sorted(d for d in root.glob(args.pattern) if d.is_dir())
    if not dirs:
        raise SystemExit(f"No directories matching {root}/{args.pattern}")

    rows = [r for r in (collect_camera(d) for d in dirs) if r is not None]
    if not rows:
        raise SystemExit(f"None of {dirs} contain fold*_preds.pkl files yet.")

    lines = []
    lines.append("=" * 78)
    lines.append("CROSS-CAMERA LOO-CV COMPARISON")
    lines.append("=" * 78)
    lines.append(
        f"{'camera':>12} {'folds':>6} {'mean ± std':>18} "
        f"{'pooled':>9} {'macro F1':>10} {'frames':>9}"
    )
    for r in rows:
        ms = f"{r['mean_acc']:.3f} ± {r['std_acc_sample']:.3f}"
        lines.append(
            f"{r['camera']:>12} {r['n_folds']:>6} {ms:>18} "
            f"{r['pooled_acc']:>9.3f} {r['macro_f1']:>10.3f} {r['n_frames']:>9}"
        )
    lines.append("\n  (std = sample std, ddof=1)")

    lines.append("\nPer-fold frame accuracy:")
    header = f"{'fold':>5}" + "".join(f"{r['camera']:>12}" for r in rows)
    lines.append(header)
    for k in range(9):
        row = f"{k:>5}"
        for r in rows:
            v = r["per_fold_acc"][k]
            row += "         nan" if np.isnan(v) else f"{v:>12.3f}"
        lines.append(row)

    report = "\n".join(lines)
    print(report)
    (out / "cameras_comparison.txt").write_text(report + "\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(9)
    n_cam = len(rows)
    w = 0.8 / max(n_cam, 1)
    for i, r in enumerate(rows):
        offset = (i - (n_cam - 1) / 2) * w
        accs = np.array(r["per_fold_acc"])
        ax.bar(x + offset, accs, w, label=r["camera"])
    ax.set_xticks(x)
    ax.set_xlabel("held-out video (fold)")
    ax.set_ylabel("frame accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("LOO-CV per-fold accuracy by camera")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "cameras_comparison.png", dpi=150)
    plt.close()

    (out / "cameras_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(f"\nSaved: {out / 'cameras_comparison.txt'}")
    print(f"Saved: {out / 'cameras_comparison.png'}")
    print(f"Saved: {out / 'cameras_comparison.json'}")


# ══════════════════════════════════════════════════════════════════════
# Exp 3 — static hard/soft voting
# ══════════════════════════════════════════════════════════════════════

def static_collect_pooled(root, cams, mode, weights_map=None):
    """mode in {'soft', 'hard'}. Returns pooled metrics across all 9 folds."""
    yts, vps = [], []
    for k in range(9):
        rows = []
        for c in cams:
            d = load_fold_exact(root, c, k)
            if d is None:
                continue
            if mode == "soft" and "y_soft" not in d:
                onehot = np.zeros((len(d["y_pred"]), len(CLASSES)))
                onehot[np.arange(len(d["y_pred"])), d["y_pred"]] = 1.0
                rows.append((c, d["y_true"], d["y_pred"], onehot))
            else:
                rows.append((c, d["y_true"], d["y_pred"], d.get("y_soft")))
        if not rows:
            continue
        yt = rows[0][1]
        w = [weights_map[r[0]] for r in rows] if weights_map is not None else None
        if mode == "soft":
            yp = soft_vote([r[3] for r in rows], w)
        else:
            yp = hard_vote([r[2] for r in rows], w)
        yts.append(yt); vps.append(yp)
    if not yts:
        return None
    return pooled_metrics(np.concatenate(yts), np.concatenate(vps))


def cmd_static(args):
    root = Path(args.root)
    cameras = args.cameras

    single, has_soft = {}, {}
    for cam in cameras:
        yts, yps, has = [], [], True
        for k in range(9):
            d = load_fold_exact(root, cam, k)
            if d is None:
                continue
            yts.append(d["y_true"]); yps.append(d["y_pred"])
            if "y_soft" not in d:
                has = False
        if yts:
            single[cam] = pooled_metrics(np.concatenate(yts), np.concatenate(yps))
        has_soft[cam] = has

    print("=" * 78)
    print(" STATIC VOTING (hard / soft) — vitrectomy LOO")
    print("=" * 78)
    print()
    print(" Per-camera (single, no ensemble):")
    print(f"   {'camera':<10s} {'acc':>9s} {'macro F1':>10s} {'n':>9s} {'y_soft':>9s}")
    for cam in cameras:
        s = single.get(cam, {})
        soft_tag = "yes" if has_soft.get(cam) else "missing"
        if s:
            print(f"   {cam:<10s} {s['acc']:>9.4f} {s['macro_f1']:>10.4f}"
                  f" {s['n']:>9d} {soft_tag:>9s}")
        else:
            print(f"   {cam:<10s}        -          -         -   no folds")

    weights = {c: single[c]["acc"] for c in cameras if c in single}

    print()
    print(f" Ensembles across all {len(cameras)} cameras:")
    for mode, label in [("hard", "hard vote"), ("soft", "soft vote")]:
        m = static_collect_pooled(root, cameras, mode)
        mw = static_collect_pooled(root, cameras, mode, weights)
        if m and mw:
            print(f"   {label + ' uniform':<24s}"
                  f"acc={m['acc']:.4f}   macroF1={m['macro_f1']:.4f}")
            print(f"   {label + ' weighted':<24s}"
                  f"acc={mw['acc']:.4f}   macroF1={mw['macro_f1']:.4f}")

    print()
    print(" Top-8 camera subsets (soft, weighted, by macro F1):")
    results = []
    for r in range(2, len(cameras) + 1):
        for sub in combinations(cameras, r):
            m = static_collect_pooled(root, list(sub), "soft", weights)
            if m:
                results.append((sub, m["acc"], m["macro_f1"]))
    results.sort(key=lambda x: -x[2])
    for sub, a, f in results[:8]:
        label = "+".join(c[-2:] for c in sub)
        print(f"   {label:<22s} acc={a:.4f}  macroF1={f:.4f}")

    out_txt = root / "soft_voting_comparison.txt"
    lines = []
    lines.append("=" * 78)
    lines.append(" STATIC VOTING (hard / soft) — vitrectomy LOO")
    lines.append("=" * 78)
    lines.append("")
    lines.append(" Single-camera pooled:")
    for cam in cameras:
        s = single.get(cam, {})
        if s:
            lines.append(f"   {cam}: acc={s['acc']:.4f}  macroF1={s['macro_f1']:.4f}  n={s['n']}")
    lines.append("")
    lines.append(" Ensembles:")
    lines.append(f"   hard vote uniform  : {static_collect_pooled(root, cameras, 'hard')}")
    lines.append(f"   hard vote weighted : {static_collect_pooled(root, cameras, 'hard', weights)}")
    lines.append(f"   soft vote uniform  : {static_collect_pooled(root, cameras, 'soft')}")
    lines.append(f"   soft vote weighted : {static_collect_pooled(root, cameras, 'soft', weights)}")
    lines.append("")
    lines.append(" Top subsets (soft, weighted, by macro F1):")
    for sub, a, f in results[:10]:
        lines.append(f"   {'+'.join(c[-2:] for c in sub):<22s} acc={a:.4f}  macroF1={f:.4f}")
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out_txt}")

    # Comparison bar chart. The "best subset" choices below are HARDCODED to
    # the ones the write-up tells the story around (01+03+05 for hard, both
    # 01+03+05 and 03+05 for soft) — only meaningful for the 5-camera setup.
    if len(cameras) == 5:
        soft5_w = static_collect_pooled(root, cameras, "soft", weights)
        hard5_w = static_collect_pooled(root, cameras, "hard", weights)
        hard_135 = static_collect_pooled(root, ["camera01", "camera03", "camera05"], "hard", weights)
        soft_135 = static_collect_pooled(root, ["camera01", "camera03", "camera05"], "soft", weights)
        soft_35 = static_collect_pooled(root, ["camera03", "camera05"], "soft", weights)
        best_single_cam = max(single, key=lambda c: single[c]["acc"])
        bs = single[best_single_cam]

        bars = [
            ("best single\n" + best_single_cam, bs["acc"], "#1f77b4"),
            ("hard vote\nweighted (5)", hard5_w["acc"], "#2ca02c"),
            ("hard vote\n01+03+05", hard_135["acc"], "#17becf"),
            ("soft vote\nweighted (5)", soft5_w["acc"], "#9467bd"),
            ("soft vote\n01+03+05", soft_135["acc"], "#e377c2"),
            ("soft vote\n03+05", soft_35["acc"], "#d62728"),
        ]
        fig, ax = plt.subplots(figsize=(11, 5))
        xs = np.arange(len(bars))
        vals = [v for _, v, _ in bars]
        cols = [c for _, _, c in bars]
        ax.bar(xs, vals, color=cols)
        ax.set_xticks(xs)
        ax.set_xticklabels([n for n, _, _ in bars], fontsize=9)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
        ax.axhline(bs["acc"], ls='--', color='blue', alpha=0.5,
                   label=f"best single = {bs['acc']:.3f}")
        ax.set_ylim(0.79, 0.90)
        ax.set_ylabel("pooled frame accuracy")
        ax.set_title("Hard vs soft voting (pooled accuracy across 9 LOO folds)")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        plt.savefig(root / "soft_voting_comparison.png", dpi=150)
        print(f"Saved: {root / 'soft_voting_comparison.png'}")


# ══════════════════════════════════════════════════════════════════════
# Exp 4 — dynamic entropy top-k voting
# ══════════════════════════════════════════════════════════════════════

def cmd_dynamic(args):
    root = Path(args.root)
    out = Path(args.out) if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_ood" if args.ood else ""

    windows = []
    for w in args.windows:
        windows.append(float("inf") if w.lower() in ("inf", "video") else int(w))

    missing = []
    for cam in args.cameras:
        for k in args.folds:
            d = load_fold_by_pattern(root, args.pattern, cam, k, args.ood)
            if d is None:
                missing.append(f"{cam}/fold{k}")
            elif "y_soft" not in d:
                missing.append(f"{cam}/fold{k}: no y_soft")
    if missing:
        print("WARNING: missing or hard-only folds:")
        for m in missing[:20]:
            print(f"   {m}")
        if len(missing) > 20:
            print(f"   ... ({len(missing) - 20} more)")
        print("Re-run loo_evaluate.py first.")

    per_cam_pooled = {}
    for cam in args.cameras:
        yts, yps = [], []
        for k in args.folds:
            d = load_fold_by_pattern(root, args.pattern, cam, k, args.ood)
            if d is None:
                continue
            yts.append(d["y_true"]); yps.append(d["y_pred"])
        if yts:
            yt = np.concatenate(yts); yp = np.concatenate(yps)
            per_cam_pooled[cam] = (
                float((yt == yp).mean()),
                float(f1_score(yt, yp, labels=list(range(len(CLASSES))),
                               average="macro", zero_division=0))
            )

    pooled_results = {}
    per_fold = {w: [] for w in windows}
    per_fold_seg = {w: [] for w in windows}
    for w in windows:
        yts, yps = [], []
        selection_counts = np.zeros(len(args.cameras))
        for k in args.folds:
            per_cam_soft, yt = {}, None
            for cam in args.cameras:
                d = load_fold_by_pattern(root, args.pattern, cam, k, args.ood)
                if d is None or "y_soft" not in d:
                    continue
                per_cam_soft[cam] = np.asarray(d["y_soft"])
                yt = np.asarray(d["y_true"])
            if not per_cam_soft or yt is None:
                continue
            yp, sel, _ = dynamic_top_k_vote(per_cam_soft, w, k=args.k)
            yts.append(yt); yps.append(yp)
            per_fold[w].append((k, float((yp == yt).mean()), len(yt)))
            per_fold_seg[w].append((k, edit_score(yp, yt),
                                    f1_at_k(yp, yt, 0.10),
                                    f1_at_k(yp, yt, 0.25),
                                    f1_at_k(yp, yt, 0.50)))
            for ci, cam in enumerate(args.cameras):
                if cam in per_cam_soft:
                    selection_counts[ci] += sel[:, list(per_cam_soft.keys()).index(cam)].sum()
        if yts:
            yt = np.concatenate(yts); yp = np.concatenate(yps)
            pooled_results[w] = {
                "acc": float((yt == yp).mean()),
                "macro_f1": float(f1_score(yt, yp, labels=list(range(len(CLASSES))),
                                           average="macro", zero_division=0)),
                "n": len(yt),
                "selection_counts": selection_counts.tolist(),
            }

    lines = []
    lines.append("=" * 78)
    head = "DYNAMIC TOP-K SOFT VOTING"
    if args.ood:
        head += "  (OOD evaluation)"
    lines.append(head)
    lines.append("=" * 78)
    lines.append(f"  cameras  : {args.cameras}")
    lines.append(f"  folds    : {args.folds}")
    lines.append(f"  k        : {args.k}  (top cameras vote per frame)")
    lines.append(f"  windows  : {args.windows}  "
                 f"(0 = strict per-frame, N = ±N s smoothing, inf = per-video)")
    lines.append("")
    lines.append("Single-camera baselines (pooled, no ensemble):")
    for cam in args.cameras:
        if cam in per_cam_pooled:
            a, f = per_cam_pooled[cam]
            lines.append(f"   {cam:<10s} acc={a:.4f}  macroF1={f:.4f}")
    best_solo = max(per_cam_pooled, key=lambda c: per_cam_pooled[c][0])
    lines.append(f"  → best single: {best_solo} "
                 f"({per_cam_pooled[best_solo][0]:.4f} acc)")
    lines.append("")
    lines.append("Dynamic Top-K voting (pooled):")
    lines.append(f"  {'window':>12} {'acc':>8} {'macroF1':>10} {'frames':>9}  selection rate per camera")
    for w_str, w in zip(args.windows, windows):
        r = pooled_results.get(w)
        if not r:
            continue
        sel = np.array(r["selection_counts"])
        sel_pct = sel / max(sel.sum(), 1) * 100
        sel_repr = "  ".join(f"{cam[-2:]}:{s:>5.1f}%" for cam, s in zip(args.cameras, sel_pct))
        lines.append(f"  {w_str:>12} {r['acc']:>8.4f} {r['macro_f1']:>10.4f} {r['n']:>9}  {sel_repr}")

    lines.append("")
    lines.append("Per-fold accuracy by window:")
    cols = "fold " + " ".join(f"{w:>10s}" for w in args.windows)
    lines.append(f"  {cols}")
    fold_set = sorted({k for w in windows for (k, _, _) in per_fold[w]})
    for k in fold_set:
        row = f"  {k:>4}"
        for w in windows:
            entry = next((a for (kk, a, _) in per_fold[w] if kk == k), None)
            row += f" {entry:>10.4f}" if entry is not None else "        nan"
        lines.append(row)

    lines.append("")
    lines.append("Segmental metrics (Lea et al. 2017), mean over folds:")
    lines.append(f"  {'window':>12} {'Edit':>7} {'F1@10':>7} {'F1@25':>7} {'F1@50':>7}  {'folds':>5}")
    for w_str, w in zip(args.windows, windows):
        rows = per_fold_seg.get(w, [])
        if not rows:
            continue
        arr = np.array([(e, a, b, c) for _, e, a, b, c in rows])
        m = arr.mean(axis=0)
        lines.append(f"  {w_str:>12} {m[0]:>7.2f} {m[1]:>7.2f} {m[2]:>7.2f} {m[3]:>7.2f}  {len(rows):>5}")
    lines.append("  (window 0 = 'Dynamic Top-3, per-frame'; "
                 "inf = 'Dynamic Top-3, per-video')")

    report = "\n".join(lines)
    print(report)
    out_txt = out / f"dynamic_voting{suffix}.txt"
    out_txt.write_text(report + "\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"best single\n({best_solo})"]
    vals = [per_cam_pooled[best_solo][0]]
    for w_str, w in zip(args.windows, windows):
        if w in pooled_results:
            labels.append(f"Δ = {w_str}\ntop-{args.k}")
            vals.append(pooled_results[w]["acc"])
    xs = np.arange(len(vals))
    ax.bar(xs, vals, color=["#1f77b4"] + ["#d62728"] * (len(vals) - 1))
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("pooled frame accuracy")
    ax.set_ylim(0.7, 1.0)
    ax.set_title(f"Dynamic top-{args.k} soft voting" + (" — OOD" if args.ood else ""))
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    out_png = out / f"dynamic_voting{suffix}.png"
    plt.savefig(out_png, dpi=150)
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_png}")


# ══════════════════════════════════════════════════════════════════════
# Exp 5 — per-frame vs per-video diagnostics
# ══════════════════════════════════════════════════════════════════════

def cmd_pf_vs_pv(args):
    root = Path(args.root)
    suffix = "_ood" if args.ood else ""
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    fold_data = []
    for k in args.folds:
        per_cam_soft, yt = {}, None
        for cam in args.cameras:
            d = load_fold_by_pattern(root, args.pattern, cam, k, args.ood)
            if d is None or "y_soft" not in d:
                continue
            per_cam_soft[cam] = np.asarray(d["y_soft"])
            yt = np.asarray(d["y_true"])
        if not per_cam_soft or yt is None:
            continue
        yp_pf, sel_pf, H = dynamic_top_k_vote(per_cam_soft, 0, k=args.k)
        yp_pv, sel_pv, _ = dynamic_top_k_vote(per_cam_soft, float("inf"), k=args.k)
        fold_data.append({
            "fold": k, "y_true": yt,
            "yp_pf": yp_pf, "yp_pv": yp_pv,
            "sel_pf": sel_pf, "sel_pv": sel_pv,
            "H": H,
            "cams_used": list(per_cam_soft.keys()),
        })

    if not fold_data:
        raise SystemExit(f"No folds with y_soft found under {root}/{args.pattern}")

    cams_used = fold_data[0]["cams_used"]

    say("=" * 78)
    say(" PER-FRAME vs PER-VIDEO  —  diagnostic comparison")
    if args.ood:
        say(" (OOD evaluation)")
    say("=" * 78)
    say(f"  pattern: {args.pattern}")
    say(f"  cams   : {cams_used}")
    say(f"  folds  : {args.folds}")
    say(f"  k      : {args.k}")
    say("")

    yt_all = np.concatenate([d["y_true"] for d in fold_data])
    yp_pf_all = np.concatenate([d["yp_pf"] for d in fold_data])
    yp_pv_all = np.concatenate([d["yp_pv"] for d in fold_data])

    def metric_block(name):
        key = "yp_pf" if name == "pf" else "yp_pv"
        yp_all = yp_pf_all if name == "pf" else yp_pv_all
        acc = float((yt_all == yp_all).mean())
        f1 = float(f1_score(yt_all, yp_all, labels=list(range(len(CLASSES))),
                            average="macro", zero_division=0))
        ed = np.mean([edit_score(d[key], d["y_true"]) for d in fold_data])
        f10 = np.mean([f1_at_k(d[key], d["y_true"], 0.10) for d in fold_data])
        f25 = np.mean([f1_at_k(d[key], d["y_true"], 0.25) for d in fold_data])
        f50 = np.mean([f1_at_k(d[key], d["y_true"], 0.50) for d in fold_data])
        return acc, f1, ed, f10, f25, f50

    pf = metric_block("pf")
    pv = metric_block("pv")

    say("(1) Top-line metrics (pooled across folds):")
    say(f"   {'strategy':<14s} {'acc':>8s} {'macroF1':>9s} {'Edit':>8s} {'F1@10':>8s} {'F1@25':>8s} {'F1@50':>8s}")
    say(f"   {'per-frame (Δ=0)':<14s} {pf[0]:>8.4f} {pf[1]:>9.4f} {pf[2]:>8.2f} {pf[3]:>8.2f} {pf[4]:>8.2f} {pf[5]:>8.2f}")
    say(f"   {'per-video (Δ=∞)':<14s} {pv[0]:>8.4f} {pv[1]:>9.4f} {pv[2]:>8.2f} {pv[3]:>8.2f} {pv[4]:>8.2f} {pv[5]:>8.2f}")
    say(f"   {'Δ (pv - pf)':<14s} {pv[0]-pf[0]:>+8.4f} {pv[1]-pf[1]:>+9.4f} {pv[2]-pf[2]:>+8.2f} {pv[3]-pf[3]:>+8.2f} {pv[4]-pf[4]:>+8.2f} {pv[5]-pf[5]:>+8.2f}")
    say("")

    say("(2) Per-fold accuracy & winner:")
    say(f"   {'fold':>4} {'pf_acc':>8} {'pv_acc':>8} {'Δ pf-pv':>9} {'winner':>8}")
    pf_wins = pv_wins = ties = 0
    for d in fold_data:
        a_pf = float((d["yp_pf"] == d["y_true"]).mean())
        a_pv = float((d["yp_pv"] == d["y_true"]).mean())
        if abs(a_pf - a_pv) < 1e-4:
            w = "tie"; ties += 1
        elif a_pf > a_pv:
            w = "pf"; pf_wins += 1
        else:
            w = "pv"; pv_wins += 1
        say(f"   {d['fold']:>4} {a_pf:>8.4f} {a_pv:>8.4f} {a_pf-a_pv:>+9.4f} {w:>8s}")
    say(f"\n   winner count: per-frame={pf_wins}  per-video={pv_wins}  ties={ties}")
    say("")

    diff_mask = yp_pf_all != yp_pv_all
    n_diff = int(diff_mask.sum())
    if n_diff > 0:
        pf_right_pv_wrong = int(((yp_pf_all == yt_all) & (yp_pv_all != yt_all)).sum())
        pv_right_pf_wrong = int(((yp_pv_all == yt_all) & (yp_pf_all != yt_all)).sum())
        both_wrong = int(((yp_pf_all != yt_all) & (yp_pv_all != yt_all) & diff_mask).sum())
        say("(3) Disagreement analysis "
            f"({n_diff} frames / {n_diff/len(yt_all)*100:.1f}% of total differ):")
        say(f"   per-frame correct, per-video wrong : {pf_right_pv_wrong:>5d}  "
            f"({pf_right_pv_wrong/n_diff*100:.1f}%)")
        say(f"   per-video correct, per-frame wrong : {pv_right_pf_wrong:>5d}  "
            f"({pv_right_pf_wrong/n_diff*100:.1f}%)")
        say(f"   both wrong (different wrong answers): {both_wrong:>5d}  "
            f"({both_wrong/n_diff*100:.1f}%)")
        net = pf_right_pv_wrong - pv_right_pf_wrong
        if net > 0:
            say(f"\n   → at disagreement, per-frame is right {net} more times. ")
        elif net < 0:
            say(f"\n   → at disagreement, per-video is right {-net} more times.")
        else:
            say("\n   → at disagreement, neither systematically better.")
        say("")
    else:
        say("(3) Disagreement analysis: pf and pv never disagree.\n")

    say("(4) Per-class F1 (pooled):")
    say(f"   {'class':<18} {'pf F1':>8} {'pv F1':>8}  {'Δ pv-pf':>9}")
    for c, name in enumerate(CLASSES):
        f_pf = f1_score(yt_all == c, yp_pf_all == c, zero_division=0)
        f_pv = f1_score(yt_all == c, yp_pv_all == c, zero_division=0)
        say(f"   {name:<18} {f_pf:>8.3f} {f_pv:>8.3f}  {f_pv-f_pf:>+9.3f}")
    say("")

    say("(5) Camera selection rate (% of frames each camera was in top-k):")
    say(f"   {'camera':<10} {'pf rate':>9} {'pv rate':>9}")
    n_cams = len(cams_used)
    pf_counts = np.zeros(n_cams); pv_counts = np.zeros(n_cams)
    total = 0
    for d in fold_data:
        pf_counts += d["sel_pf"].sum(axis=0)
        pv_counts += d["sel_pv"].sum(axis=0)
        total += d["sel_pf"].shape[0]
    for ci, cam in enumerate(cams_used):
        say(f"   {cam:<10} {pf_counts[ci]/total*100:>8.1f}% {pv_counts[ci]/total*100:>8.1f}%")
    say("")

    say("(6) Per-frame selection switch rate:")
    say("   (fraction of frame-pairs (t, t+1) where the top-3 set changed)")
    say(f"   {'fold':>4} {'switch%':>9}")
    rates = []
    for d in fold_data:
        r = selection_switch_rate(d["sel_pf"])
        rates.append(r)
        say(f"   {d['fold']:>4} {r*100:>8.1f}%")
    say(f"   {'mean':>4} {np.mean(rates)*100:>8.1f}%")
    say(f"   {'std':>4} {np.std(rates, ddof=1)*100:>8.1f}%")
    say("")

    say(f"(7) Boundary (±{args.boundary_window} frames around phase change) "
        "vs interior accuracy:")
    bd_pf_t, bd_pf_p, in_pf_t, in_pf_p = [], [], [], []
    bd_pv_t, bd_pv_p, in_pv_t, in_pv_p = [], [], [], []
    for d in fold_data:
        mask = boundary_mask(d["y_true"], window=args.boundary_window)
        bd_pf_t.append(d["y_true"][mask]); bd_pf_p.append(d["yp_pf"][mask])
        bd_pv_t.append(d["y_true"][mask]); bd_pv_p.append(d["yp_pv"][mask])
        in_pf_t.append(d["y_true"][~mask]); in_pf_p.append(d["yp_pf"][~mask])
        in_pv_t.append(d["y_true"][~mask]); in_pv_p.append(d["yp_pv"][~mask])
    bd_pf_t = np.concatenate(bd_pf_t); bd_pf_p = np.concatenate(bd_pf_p)
    in_pf_t = np.concatenate(in_pf_t); in_pf_p = np.concatenate(in_pf_p)
    bd_pv_t = np.concatenate(bd_pv_t); bd_pv_p = np.concatenate(bd_pv_p)
    in_pv_t = np.concatenate(in_pv_t); in_pv_p = np.concatenate(in_pv_p)
    say(f"   boundary frames: {len(bd_pf_t):>5d}")
    say(f"     per-frame  acc on boundary : {float((bd_pf_p==bd_pf_t).mean()):.4f}")
    say(f"     per-video  acc on boundary : {float((bd_pv_p==bd_pv_t).mean()):.4f}")
    say(f"   interior frames: {len(in_pf_t):>5d}")
    say(f"     per-frame  acc on interior : {float((in_pf_p==in_pf_t).mean()):.4f}")
    say(f"     per-video  acc on interior : {float((in_pv_p==in_pv_t).mean()):.4f}")
    say("")
    say("   → If per-frame helps anywhere, it should help on boundaries "
        "(different cams may see different phases at the transition).")
    say("")

    say("(8) Confidence (mean entropy of selected cameras) vs correctness:")
    pf_H_per_frame, pv_H_per_frame = [], []
    for d in fold_data:
        H = d["H"]
        sel_pf, sel_pv = d["sel_pf"], d["sel_pv"]
        h_pf = (H.T * sel_pf).sum(axis=1) / sel_pf.sum(axis=1)
        h_pv = (H.T * sel_pv).sum(axis=1) / sel_pv.sum(axis=1)
        pf_H_per_frame.append(h_pf); pv_H_per_frame.append(h_pv)
    H_pf = np.concatenate(pf_H_per_frame)
    H_pv = np.concatenate(pv_H_per_frame)
    right_pf = yp_pf_all == yt_all
    right_pv = yp_pv_all == yt_all
    say(f"   per-frame: mean H on correct = {H_pf[right_pf].mean():.3f}, "
        f"on wrong = {H_pf[~right_pf].mean():.3f}, gap = "
        f"{H_pf[~right_pf].mean() - H_pf[right_pf].mean():+.3f}")
    say(f"   per-video: mean H on correct = {H_pv[right_pv].mean():.3f}, "
        f"on wrong = {H_pv[~right_pv].mean():.3f}, gap = "
        f"{H_pv[~right_pv].mean() - H_pv[right_pv].mean():+.3f}")
    say("   → larger gap = better-calibrated ensemble (low confidence on "
        "errors, high on hits).")
    say("")

    out_txt = root / f"pf_vs_pv{suffix}.txt"
    out_txt.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    folds = [d["fold"] for d in fold_data]
    pf_a = [float((d["yp_pf"] == d["y_true"]).mean()) for d in fold_data]
    pv_a = [float((d["yp_pv"] == d["y_true"]).mean()) for d in fold_data]
    w = 0.35
    ax.bar(np.arange(len(folds)) - w/2, pf_a, w, label="per-frame (Δ=0)", color="#d62728")
    ax.bar(np.arange(len(folds)) + w/2, pv_a, w, label="per-video (Δ=∞)", color="#2ca02c")
    ax.set_xticks(np.arange(len(folds)))
    ax.set_xticklabels(folds)
    ax.set_xlabel("fold"); ax.set_ylabel("frame accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Per-fold accuracy: per-frame vs per-video")
    ax.legend(loc="lower right")
    for x, (a, b) in enumerate(zip(pf_a, pv_a)):
        ax.text(x - w/2, a + 0.005, f"{a:.2f}", ha="center", fontsize=7)
        ax.text(x + w/2, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)

    ax = axes[1]
    xs = np.arange(n_cams)
    ax.bar(xs - 0.2, pf_counts/total*100, 0.4, label="per-frame", color="#d62728")
    ax.bar(xs + 0.2, pv_counts/total*100, 0.4, label="per-video", color="#2ca02c")
    ax.set_xticks(xs)
    ax.set_xticklabels([c[-2:] for c in cams_used])
    ax.set_xlabel("camera"); ax.set_ylabel("% of frames in top-k")
    ax.set_title("Camera selection rate")
    ax.legend()
    plt.tight_layout()
    out_png = root / f"pf_vs_pv{suffix}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_png}")


# ══════════════════════════════════════════════════════════════════════
# Exp 7 — sensor-dropout robustness
# ══════════════════════════════════════════════════════════════════════

def per_fold_predictions(root, pattern, cameras, fold, ood):
    softs, yt = {}, None
    for cam in cameras:
        d = load_fold_by_pattern(root, pattern, cam, fold, ood)
        if d is None or "y_soft" not in d:
            continue
        softs[cam] = np.asarray(d["y_soft"])
        yt = np.asarray(d["y_true"])
    return yt, softs


def cmd_dropout(args):
    root = Path(args.root)
    suffix = "_ood" if args.ood else ""
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    cam_pool = {}
    for cam in args.cameras:
        yts, yps = [], []
        for k in args.folds:
            d = load_fold_by_pattern(root, args.pattern, cam, k, args.ood)
            if d:
                yts.append(d["y_true"]); yps.append(d["y_pred"])
        if yts:
            yt = np.concatenate(yts); yp = np.concatenate(yps)
            cam_pool[cam] = float((yt == yp).mean())

    say("=" * 78)
    say(" SENSOR DROPOUT ROBUSTNESS")
    if args.ood:
        say(" (OOD evaluation)")
    say("=" * 78)
    say("")
    say(f"  cameras: {args.cameras}")
    say(f"  folds  : {args.folds}")
    say("  per-camera pooled (used as soft-vote weights):")
    for c, a in cam_pool.items():
        say(f"     {c}: {a:.4f}")
    say("")

    say("Scenario A — drop the 2 most-confident cameras per video,")
    say("            vote with the remaining 3:")
    say("")
    say(f"   {'fold':<6} {'video / run':<25} {'full 5':>8} {'after drop':>11} {'Δ':>6}   dropped")

    full_yts, full_yps, drop_yts, drop_yps = [], [], [], []
    for k in args.folds:
        yt, softs = per_fold_predictions(root, args.pattern, args.cameras, k, args.ood)
        if yt is None or len(softs) < 3:
            continue
        full_pred = confidence_soft_vote(softs, weights=cam_pool)
        full_acc = float((full_pred == yt).mean())

        entropies = {c: softmax_entropy(s).mean() for c, s in softs.items()}
        confident_order = sorted(entropies, key=entropies.get)
        drop = sorted(confident_order[:2])
        remain = {c: s for c, s in softs.items() if c not in drop}
        drop_pred = confidence_soft_vote(remain, weights=cam_pool)
        drop_acc = float((drop_pred == yt).mean())

        label = RUN_LABELS.get(k, f"video_{k}")
        say(f"   {k:<6} {label:<25} {full_acc:>8.3f} {drop_acc:>11.3f}  "
            f"{drop_acc - full_acc:>+6.3f}   {drop}")
        full_yts.append(yt); full_yps.append(full_pred)
        drop_yts.append(yt); drop_yps.append(drop_pred)

    fy = np.concatenate(full_yts); fp = np.concatenate(full_yps)
    dy = np.concatenate(drop_yts); dp = np.concatenate(drop_yps)
    full_pool = float((fp == fy).mean())
    full_f1 = f1_score(fy, fp, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    drop_pool = float((dp == dy).mean())
    drop_f1 = f1_score(dy, dp, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    say("")
    say(f"   pooled full-5  : acc={full_pool:.4f}  macroF1={full_f1:.4f}")
    say(f"   pooled after A : acc={drop_pool:.4f}  macroF1={drop_f1:.4f}")
    say(f"   net Δ acc      : {drop_pool - full_pool:+.4f}")

    say("")
    say("Scenario B — exhaustive pair-drop sweep (vote with remaining 3):")
    say("")
    pair_pooled = []
    for drop in combinations(args.cameras, 2):
        keep = [c for c in args.cameras if c not in drop]
        yts, yps = [], []
        for k in args.folds:
            yt, softs = per_fold_predictions(root, args.pattern, keep, k, args.ood)
            if yt is None or not softs:
                continue
            yp = confidence_soft_vote(softs, weights=cam_pool)
            yts.append(yt); yps.append(yp)
        if yts:
            yt = np.concatenate(yts); yp = np.concatenate(yps)
            pair_pooled.append((drop, float((yt == yp).mean())))

    pair_pooled.sort(key=lambda x: -x[1])
    say(f"   {'dropped pair':<32} {'pooled acc':>11}")
    for drop, a in pair_pooled:
        say(f"   {' + '.join(drop):<32} {a:>11.4f}")
    accs = np.array([a for _, a in pair_pooled])
    say("")
    say(f"   best-case (luckiest drop) : {accs.max():.4f}")
    say(f"   worst-case (worst drop)   : {accs.min():.4f}")
    say(f"   mean over all 10 drops    : {accs.mean():.4f}")
    say(f"   std (sample)              : {accs.std(ddof=1):.4f}")
    say(f"   full 5 baseline           : {full_pool:.4f}")
    say("")
    say(f"  → Any 2-camera failure keeps acc within "
        f"[{accs.min():.3f}, {accs.max():.3f}] — quantifies sensor resilience.")

    out_txt = root / f"sensor_dropout{suffix}.txt"
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {out_txt}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare-cameras", help="Exp 2 — per-camera accuracy, no voting.")
    p.add_argument("--root", default="outputs", type=str)
    p.add_argument("--pattern", default="loo_cv_camera*", type=str)
    p.add_argument("--out", default=None, type=str)
    p.set_defaults(func=cmd_compare_cameras)

    p = sub.add_parser("static", help="Exp 3 — hard vs soft vote, brute-force subsets.")
    p.add_argument("--root", default="outputs", type=str)
    p.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    p.set_defaults(func=cmd_static)

    p = sub.add_parser("dynamic", help="Exp 4 — entropy-ranked top-k vote.")
    p.add_argument("--root", default="outputs", type=str)
    p.add_argument("--pattern", default="loo_cv_camera*", type=str,
                    help="Glob for per-camera result dirs (e.g. "
                         "'loo_cv_camera*' or 'expAB_loo_cv_camera*').")
    p.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(9)))
    p.add_argument("--k", type=int, default=3,
                    help="Top-K cameras to vote per frame (default 3).")
    p.add_argument("--windows", nargs="+", type=str, default=["0", "5", "inf"],
                    help="Smoothing window sizes in frames (1 frame = 1 s "
                         "at 1 fps). 'inf' = per-video. Default: 0 5 inf.")
    p.add_argument("--ood", action="store_true",
                    help="Use fold<k>_preds_ood.pkl (Experiment B).")
    p.add_argument("--out", default=None, type=str, help="Output dir; default = --root.")
    p.set_defaults(func=cmd_dynamic)

    p = sub.add_parser("pf-vs-pv", help="Exp 5 — per-frame vs per-video diagnostics.")
    p.add_argument("--root", default="outputs", type=str)
    p.add_argument("--pattern", default="loo_cv_camera*", type=str)
    p.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(9)))
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--ood", action="store_true")
    p.add_argument("--boundary_window", type=int, default=3,
                    help="Frames (=seconds at 1 fps) around a true phase "
                         "change to count as 'boundary'.")
    p.set_defaults(func=cmd_pf_vs_pv)

    p = sub.add_parser("dropout", help="Exp 7 — sensor-dropout robustness.")
    p.add_argument("--root", default="outputs", type=str)
    p.add_argument("--pattern", default="loo_cv_camera*", type=str)
    p.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(9)))
    p.add_argument("--ood", action="store_true")
    p.set_defaults(func=cmd_dropout)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
