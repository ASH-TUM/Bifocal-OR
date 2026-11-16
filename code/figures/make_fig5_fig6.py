"""Bifocal-OR Figs. 5 and 6 from the saved test predictions (no training, no re-run).

Fig. 5: row-normalized confusion matrix of cam03, pooled over the 8 test folds (Experiment A).
Fig. 6: phase timelines of one run: ground truth, cam03, and the per-video dynamic top-3 ensemble.

Inputs: the prediction files written by evaluation/loo_evaluate.py,
    <predictions>/camera0X/foldK_preds.pkl      (or <predictions>/loo_cv_camera0X/foldK_preds.pkl)
each holding y_true, y_pred, y_soft for test fold K (fold K = Run K+1, 1 sample per ~1 s).

With --repo the ensemble and the Edit score come straight from the phase-recognition code itself
(evaluation/loo_voting.py: dynamic_top_k_vote with window = inf, k = 3;
evaluation/loo_aggregate.py: edit_score). Without --repo, built-in copies of the same
functions are used.

Usage (Windows PowerShell, paper_style.py next to this script):
    python make_fig5_fig6.py --repo ..\\phase_recognition
    python make_fig5_fig6.py --predictions "C:\\path\\to\\predictions" --run 4

Output (in --out-dir, default: current folder):
    confusion_matrix_cam03.pdf/.png   Fig. 5
    ribbon_timeline_runN.pdf/.png     Fig. 6
    fig5_fig6_check.txt               numbers to compare with the paper (pooled accuracy, Edit, segments)
    confusion_matrix_cam03.csv        counts and row percentages
"""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import paper_style as ps  # noqa: E402

CAMS = [f"camera0{i}" for i in range(1, 6)]
FOLDS = range(8)
EPS = 1e-12          # as in loo_voting.py


# ---------- built-in copies of the phase-recognition functions (used without --repo) ----------
def _entropy(p):
    p = np.clip(p, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def builtin_per_video_top3(per_cam_soft, k=3):
    """= loo_voting.dynamic_top_k_vote(per_cam_soft, window=inf, k=3)[0]"""
    softs = np.stack([per_cam_soft[c] for c in per_cam_soft])
    ent = np.stack([_entropy(s) for s in softs]).mean(axis=1)
    top = np.argsort(ent)[:k]
    w = 1.0 / np.clip(ent, EPS, None)
    avg = sum(w[c] * softs[c] for c in top) / sum(w[c] for c in top)
    return avg.argmax(axis=1)


def builtin_edit_score(pred, gt):
    """= loo_aggregate.edit_score: normalized Levenshtein on the segment label sequences."""
    P = [s[2] for s in segments(pred)]
    G = [s[2] for s in segments(gt)]
    m, n = len(P), len(G)
    D = np.zeros((m + 1, n + 1), dtype=int)
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (P[i - 1] != G[j - 1]))
    return (1 - D[m, n] / max(m, n)) * 100


def load_mateus(repo):
    """Import the voting and Edit functions from the phase-recognition code."""
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "evaluation"))
    try:
        from loo_voting import dynamic_top_k_vote
        from loo_aggregate import edit_score
    except ImportError as e:
        sys.exit(f"Could not import the phase-recognition code from {repo}: {e}\n"
                 "Install what is missing (usually: pip install scikit-learn) or run without --repo.")

    def vote(per_cam_soft):
        return dynamic_top_k_vote(per_cam_soft, float("inf"), k=3)[0]
    return vote, edit_score


# ---------- data ----------
def find_cam_dir(root, cam):
    for name in (cam, f"loo_cv_{cam}"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    sys.exit(f"No folder {cam} or loo_cv_{cam} in {root}")


def load(root, cam, fold):
    p = os.path.join(find_cam_dir(root, cam), f"fold{fold}_preds.pkl")
    if not os.path.exists(p):
        sys.exit(f"Missing: {p}")
    with open(p, "rb") as fh:
        try:
            d = pickle.load(fh)
        except ModuleNotFoundError as e:
            sys.exit(f"{e}: the predictions were saved with numpy 2. Run: pip install --upgrade \"numpy>=2\"")
    return d


def segments(y):
    y = np.asarray(y)
    out, s = [], 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[s]:
            out.append((s, i, int(y[s])))
            s = i
    return out


# ---------- Fig. 5 ----------
def fig5(root, out, log):
    M = np.zeros((5, 5), dtype=int)
    for k in FOLDS:
        d = load(root, "camera03", k)
        for t, p in zip(d["y_true"], d["y_pred"]):
            M[int(t), int(p)] += 1
    R = M / M.sum(axis=1, keepdims=True)
    log(f"Fig. 5 · cam03 pooled over folds 0-7: {M.sum()} samples, pooled accuracy "
        f"{np.trace(M) / M.sum():.4f} (paper/loo_summary: 0.900 / 0.8997)")
    log("  counts (rows = ground truth, columns = predicted):")
    for name, row in zip(ps.PHASE_NAMES, M):
        log(f"  {name:<16}" + "".join(f"{v:>7}" for v in row))
    log("  recall (diagonal, %): " + ", ".join(f"{n} {100 * R[i, i]:.1f}" for i, n in enumerate(ps.PHASE_NAMES)))
    with open(out + ".csv", "w", encoding="utf-8") as fh:
        fh.write("true\\pred," + ",".join(ps.PHASE_NAMES) + "," + ",".join(f"{n} %" for n in ps.PHASE_NAMES) + "\n")
        for i, n in enumerate(ps.PHASE_NAMES):
            fh.write(n + "," + ",".join(str(v) for v in M[i]) + "," + ",".join(f"{100 * v:.2f}" for v in R[i]) + "\n")

    ps.apply(7.5)
    fig, ax = plt.subplots(figsize=(3.45, 3.15))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1)   # square-root scale, so 5-15 % confusions are visible
    ax.imshow(R, cmap=ps.SEQ, norm=norm, aspect="equal")
    for i in range(5):
        for j in range(5):
            v = 100 * R[i, j]
            txt = "0" if M[i, j] == 0 else ("<0.1" if v < 0.05 else f"{v:.1f}")
            col = "white" if norm(R[i, j]) > 0.62 else (ps.MUTED if M[i, j] == 0 else ps.INK)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=col,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(ps.PHASE_NAMES, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ps.PHASE_NAMES)
    ax.tick_params(length=0, pad=12)
    for i, c in enumerate(ps.PHASE_COLORS):     # phase color marks next to the labels
        ax.add_patch(Rectangle((-0.5 - 0.3, i - 0.17), 0.18, 0.34, color=c, clip_on=False))
        ax.add_patch(Rectangle((i - 0.17, 4.5 + 0.12), 0.34, 0.18, color=c, clip_on=False))
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.4)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Predicted phase", labelpad=3)
    ax.set_ylabel("Ground-truth phase", labelpad=3)
    fig.tight_layout(pad=0.3)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    log(f"  written {out}.pdf / .png / .csv\n")


# ---------- per-run numbers ----------
def per_run_table(root, vote, edit, log):
    log("Per run (fold k = Run k+1): accuracy / Edit / number of segments")
    log(f"  {'Run':>3}  {'GT seg':>6}   {'cam03 acc':>9} {'Edit':>6} {'seg':>4}   {'ens. acc':>8} {'Edit':>6} {'seg':>4}")
    e3, ee = [], []
    for k in FOLDS:
        ds = {c: load(root, c, k) for c in CAMS}
        y = ds["camera03"]["y_true"].astype(int)
        c3 = ds["camera03"]["y_pred"].astype(int)
        en = np.asarray(vote({c: ds[c]["y_soft"] for c in CAMS})).astype(int)
        a, b = edit(c3, y), edit(en, y)
        e3.append(a)
        ee.append(b)
        log(f"  {k + 1:>3}  {len(segments(y)):>6}   {np.mean(c3 == y):>9.3f} {a:>6.1f} {len(segments(c3)):>4}"
            f"   {np.mean(en == y):>8.3f} {b:>6.1f} {len(segments(en)):>4}")
    log(f"  mean Edit: cam03 {np.mean(e3):.2f} (paper 39.6), ensemble {np.mean(ee):.2f} (paper 66.5)\n")


# ---------- Fig. 6 ----------
def fig6(root, out, run, vote, edit, log):
    fold = run - 1
    ds = {c: load(root, c, fold) for c in CAMS}
    y = ds["camera03"]["y_true"].astype(int)
    cam3 = ds["camera03"]["y_pred"].astype(int)
    ens = np.asarray(vote({c: ds[c]["y_soft"] for c in CAMS})).astype(int)
    rows = [("(a) Ground truth", y), ("(b) Best single view (cam03)", cam3), ("(c) Dynamic voting ensemble", ens)]
    ps.apply(7.5)
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 2.35), sharex=True,
                             gridspec_kw={"hspace": 0.55, "left": 0.03, "right": 0.97, "top": 0.8, "bottom": 0.15})
    log(f"Fig. 6 · Run {run} (fold {fold}, {len(y)} samples)")
    for ax, (name, seq) in zip(axes, rows):
        segs = segments(seq)
        for s, e, c in segs:
            ax.add_patch(Rectangle((s, 0), e - s, 1, color=ps.PHASE_COLORS[c], lw=0))
        ax.set_xlim(0, len(seq))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0, 1.12, name, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        ax.text(1, 1.12, f"{len(segs)} segments", transform=ax.transAxes, fontsize=7, color=ps.MUTED,
                va="bottom", ha="right")
        ax.tick_params(axis="x", length=2)
        log(f"  {name:<30} {len(segs):>3} segments, acc {np.mean(seq == y):.3f}, Edit {edit(seq, y):.1f}")
        log("    " + ", ".join(f"{ps.PHASE_NAMES[c]} {s}-{e}" for s, e, c in segs))
    axes[-1].set_xlabel("Time (s, 1-fps samples)", labelpad=2)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ps.PHASE_COLORS]
    fig.legend(handles, ps.PHASE_NAMES, loc="upper center", ncol=5, fontsize=6.2, frameon=False,
               handlelength=0.9, handletextpad=0.35, columnspacing=0.7, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    log(f"  written {out}.pdf / .png\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="the phase-recognition folder (code/phase_recognition): use its own voting and Edit code")
    ap.add_argument("--predictions", help="predictions folder (default: <repo>\\predictions)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--run", type=int, nargs="+", default=[4], help="run(s) for Fig. 6 (default 4)")
    a = ap.parse_args()
    if not (a.repo or a.predictions):
        ap.error("give --repo or --predictions")
    pred = a.predictions or os.path.join(a.repo, "predictions")
    if not os.path.isdir(pred):
        sys.exit(f"Predictions folder not found: {pred}")
    if a.repo:
        vote, edit = load_mateus(a.repo)
        source = f"the phase-recognition code in {a.repo}"
    else:
        vote, edit = builtin_per_video_top3, builtin_edit_score
        source = "built-in copies of loo_voting.dynamic_top_k_vote / loo_aggregate.edit_score"
    os.makedirs(a.out_dir, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log(f"Predictions: {os.path.abspath(pred)}\nEnsemble and Edit: {source}\n")
    fig5(pred, os.path.join(a.out_dir, "confusion_matrix_cam03"), log)
    per_run_table(pred, vote, edit, log)
    for r in a.run:
        fig6(pred, os.path.join(a.out_dir, f"ribbon_timeline_run{r}"), r, vote, edit, log)
    with open(os.path.join(a.out_dir, "fig5_fig6_check.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Numbers saved in fig5_fig6_check.txt")


if __name__ == "__main__":
    main()
