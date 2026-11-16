"""Bifocal-OR Fig. 4: a short excerpt of the diarization task, in the style of Fig. 2 (read-only).

Rows on the audio timeline: the ambient waveform; the audited reference speakers (orange bars on
lanes, as in Fig. 2); the pyannote.audio far-field clusters (blue bars, no identity).
One PDF (and a PNG preview) per start time.

Usage (Windows PowerShell, from the kit folder; paper_style.py must sit next to it):
    python make_fig4.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 7 --start 220

    --start   one or more start times in seconds (audio time); --dur window length (default 45)
Output: fig4_runNN_<start>.pdf / .png in the current folder.
Needs numpy and matplotlib.
"""
import argparse
import csv
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import paper_style as ps  # noqa: E402

ROLES = ["Surgeon", "OTA", "Runner", "Anesthesiologist", "Nurse", "Spectator"]
SHORT = {"Anesthesiologist": "Anesthesiologist"}
CLUSTER_COLORS = [ps.DARK_BLUE, ps.BLUE, ps.TEAL, ps.shade(ps.DARK_BLUE, 1.5), ps.shade(ps.BLUE, 1.4)]


def segments(path):
    return [(float(r["start_s"]), float(r["end_s"]), r["speaker"]) for r in csv.DictReader(open(path, encoding="utf-8"))]


def envelope(path, t0, t1, bins):
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if width != 2:
            sys.exit(f"{path}: only 16-bit PCM is supported")
        a, b = int(t0 * sr), min(n, int(t1 * sr))
        w.setpos(a)
        x = np.frombuffer(w.readframes(b - a), "<i2").astype(np.float32).reshape(-1, ch).mean(1) / 32768
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    lo = np.array([x[i:j].min() if j > i else 0 for i, j in zip(edges[:-1], edges[1:])])
    hi = np.array([x[i:j].max() if j > i else 0 for i, j in zip(edges[:-1], edges[1:])])
    return np.linspace(0, t1 - t0, bins), lo, hi


def lanes(ax, names, items, dur, t0, colors):
    """Thin lane per name (top to bottom) with rounded bars, Fig. 2 style."""
    n = len(names)
    for i, name in enumerate(names):
        y = n - 1 - i
        ax.plot([0, dur], [y, y], color=ps.LANE, lw=0.6, zorder=1)
        for a, b, s in items:
            if s == name:
                ps.bar(ax, max(a, t0) - t0, min(b, t0 + dur) - t0, y, colors[i % len(colors)])
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([SHORT.get(nm, nm) for nm in names], fontsize=7, color=ps.MUTED)
    ax.set_ylim(-0.7, max(n, 1) - 0.3)
    ax.tick_params(axis="y", length=0, pad=4)


def figure(room_wav, ref, hyp, run, t0, dur, out):
    t1 = t0 + dur
    ref = [s for s in ref if s[1] > t0 and s[0] < t1]
    hyp = [s for s in hyp if s[1] > t0 and s[0] < t1]
    roles = [r for r in ROLES if any(s[2] == r for s in ref)]
    clusters = sorted({s[2] for s in hyp})
    ps.apply(7.5)
    fig = plt.figure(figsize=(7.16, 2.75))
    hr = [0.9, 0.24 * max(len(roles), 1), 0.24 * max(len(clusters), 1)]
    gs = fig.add_gridspec(3, 1, height_ratios=hr, hspace=0.32, left=0.2, right=0.985, top=0.98, bottom=0.14)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))

    t, lo, hi = envelope(room_wav, t0, t1, 2400)
    ax0.fill_between(t, lo, hi, color=ps.DARK_BLUE, lw=0)
    m = max(abs(lo).max(), abs(hi).max(), 1e-3)
    ax0.set_ylim(-m * 1.05, m * 1.05)
    ax0.set_yticks([])

    lanes(ax1, roles, ref, dur, t0, [ps.SPEECH])
    lanes(ax2, [c for c in clusters], hyp, dur, t0, CLUSTER_COLORS)
    ax2.set_yticklabels([f"Cluster {i + 1}" for i in range(len(clusters))], fontsize=7, color=ps.MUTED)
    if not clusters:
        ax2.text(0.5, 0.5, "no speech detected", transform=ax2.transAxes, ha="center", va="center", color=ps.MUTED)

    for ax, label in ((ax0, "Ambient"), (ax1, "Reference"), (ax2, "pyannote")):
        ax.set_xlim(0, dur)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_visible(ax is ax2)
        ax.tick_params(axis="x", length=0 if ax is not ax2 else 2.5)
        if ax is not ax2:
            ax.set_xticklabels([])
        ax.grid(axis="x", color=ps.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        box = ax.get_position()
        ps.pill(ax, 0.055, box.y0 + box.height / 2, label, transform=fig.transFigure, size=7)
    # dashed separators between the groups, as between Phase and Speaker in Fig. 2
    for ax in (ax1, ax2):
        ax.annotate("", xy=(1.0, 1.06), xytext=(-0.24, 1.06), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", ls=(0, (4, 3)), color="#9AA3AB", lw=0.7))
    ax2.set_xlabel(f"Time (s) from {t0:.0f} s of the audio timeline, Run {run}", labelpad=2)
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=220)
    plt.close(fig)
    print(f"Written: {out}.pdf / .png  ({len(ref)} reference segments, {len(roles)} speakers; "
          f"pyannote: {len(hyp)} segments, {len(clusters)} clusters)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True, help="release metadata folder (Bifocal-OR)")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--start", type=float, nargs="+", required=True)
    ap.add_argument("--dur", type=float, default=45.0)
    a = ap.parse_args()
    room = os.path.join(a.root, "Ophthalmo_4D_DVR_Sync", f"Run_{a.run}", f"Run_{a.run}_Room.wav")
    ref_p = os.path.join(a.labels, "annotations", "speakers", "audited", f"run{a.run:02d}.csv")
    hyp_p = os.path.join(a.labels, "annotations", "baselines", "pyannote_ambient", f"run{a.run:02d}.csv")
    for p in (room, ref_p, hyp_p):
        if not os.path.exists(p):
            sys.exit(f"Not found: {p}")
    ref, hyp = segments(ref_p), segments(hyp_p)
    for t0 in a.start:
        figure(room, ref, hyp, a.run, t0, a.dur, f"fig4_run{a.run:02d}_{t0:.0f}")


if __name__ == "__main__":
    main()
