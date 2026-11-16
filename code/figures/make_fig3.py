"""Bifocal-OR Fig. 3: room cameras and microscope at matched times, in the paper's colors (read-only).

Three modes, all with the exact frame times and offsets of the release metadata:

  --sheet T0 T1     find the moments: room camera (top) and microscope (bottom) every --step seconds
  --views T         pick the camera: all five room cameras and the microscope at one time T
  --times T1 T2 ..  the figure: one column per time (room above microscope), a phase strip on top
                    with the moments marked (a), (b), ..., short tags under the columns (--tags)

Usage (Windows PowerShell, from the kit folder; needs bifocal_viewer.py and paper_style.py next to it):
    python make_fig3.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --sheet 1080 1110 --step 3
    python make_fig3.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --views 1190
    python make_fig3.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --camera camera01 ^
        --times 1100 1130 1142 1190 --tags "Surgeon seated" "Microscope over the eye" "Lights off" "Trocar placement"

    --camera   room camera for --sheet/--times (default camera03)
    --crop X0 Y0 X1 Y1   keep only this rectangle of the room image: left, top, right, bottom edge in
                         pixels of the full 1920x1080 frame (0,0 = top-left corner). --views draws a
                         pixel ruler, and shows the rectangle in yellow when --crop is given.
    --window S           seconds shown in the phase strip before the first and after the last time (default 25)
Times are seconds on the room-camera timeline (the timeline of the phase labels).
Output: fig3_runNN_sheet_<T0>.png, fig3_runNN_views_<T>.png, or fig3_runNN.pdf / .png.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bifocal_viewer as bv  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import paper_style as ps  # noqa: E402

# Phase palette of the Master-Video / Fig. 2 (same as bifocal_viewer.PHASE_COLORS)
PHASE_COLORS = dict(zip(ps.PHASES, ps.PHASE_COLORS))
PHASE_NAMES = {"Room_Prep": "Room Preparation", "Patient_Entry": "Patient Entry",
               "Sterile_Draping": "Sterile Draping", "Microsurgery": "Microsurgery", "Emergence": "Emergence"}
DARK_TEXT = {"Microsurgery", "Sterile_Draping"}   # light fills: dark text on them
INK = ps.INK


def setup(a):
    run_folder = bv.find_run_folder(a.root, a.run)
    phases = bv.read_phases(a.labels, a.run)
    cams = bv.RoomCameras(run_folder, phases, None, bv.read_room_times(a.labels, a.run, run_folder))
    offsets = bv.read_offsets(a.labels, a.run)
    paths = bv.find_micro(run_folder)
    keys = ["microscope"] if len(paths) == 1 else [f"microscope_part{i + 1}" for i in range(len(paths))]
    anchors = []
    for i in range(len(paths)):
        name = f"run{a.run:02d}.csv" if len(paths) == 1 else f"run{a.run:02d}_part{i + 1}.csv"
        fp = os.path.join(a.labels, "sync", "microscope_frame_times", name)
        if os.path.exists(fp):
            rows = list(csv.DictReader(open(fp, encoding="utf-8")))
            anchors.append(([int(r["frame"]) for r in rows], [float(r["t_s"]) for r in rows]))
        else:
            anchors.append(None)
    micro = bv.MicroscopeSet(paths, [offsets[k][0] if k in offsets else None for k in keys],
                             [1.0] * len(paths), anchors)
    if not micro.placed:
        sys.exit("Microscope video or its offset not found (check --labels and the run folder).")
    print(f"Run {a.run}: frame times: {'timestamps' if cams.times else 'constant rate'}; microscope: {micro.describe()}")
    return cams, micro, phases


def phase_at(phases, t):
    for s, e, p in phases:
        if s <= t <= e:
            return p
    return None


def room_image(cams, cam, t, crop, size=960):
    if cam not in cams.frames:
        return None
    k, _ = cams.frame_time(cam, t)
    img = Image.open(cams.frames[cam][1][k]).convert("RGB")
    if crop:
        img = img.crop(tuple(crop))
    img.thumbnail((size, size))
    return np.asarray(img)


def micro_image(micro, t, size=(960, 725)):
    arr, _ = micro.get(t, size)
    return arr


def show(ax, img, text="no frame"):
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor("#222222")
        ax.text(0.5, 0.5, text, color="white", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def frame(ax, color, lw=2.2):
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def sheet(a, cams, micro, phases):
    t0, t1 = a.sheet
    times = list(np.arange(t0, t1 + 1e-9, a.step))[:12]
    cols = 6
    rows = int(np.ceil(len(times) / cols))
    fig = plt.figure(figsize=(22, 6.2 * rows))
    fig.suptitle(f"Run {a.run}: {a.camera} (top) and microscope (bottom), {t0:.0f}-{times[-1]:.0f} s "
                 f"every {a.step:g} s; frame color = phase", fontsize=13)
    for i, t in enumerate(times):
        r, c = divmod(i, cols)
        ax1 = fig.add_axes([0.005 + c * 0.166, 1 - (r + 0.5) / rows + 0.03 / rows, 0.162, 0.4 / rows])
        ax2 = fig.add_axes([0.005 + c * 0.166, 1 - (r + 1) / rows + 0.04 / rows, 0.162, 0.42 / rows])
        p = phase_at(phases, t)
        show(ax1, room_image(cams, a.camera, t, a.crop))
        ax1.set_title(f"t = {t:.1f} s · {PHASE_NAMES.get(p, '-')}", fontsize=11)
        show(ax2, micro_image(micro, t), "no microscope frame")
        for ax in (ax1, ax2):
            frame(ax, PHASE_COLORS.get(p, "#999999"), 3)
    out = f"fig3_run{a.run:02d}_sheet_{t0:.0f}.png"
    fig.savefig(out, dpi=80)
    print(f"Written: {out}")


def views(a, cams, micro, phases):
    t = a.views
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.5))
    for ax, cam in zip(axes.flat, bv.CAMERAS):
        img = room_image(cams, cam, t, None, 640)
        show(ax, img)
        ax.set_title(cam, fontsize=12)
        if img is not None:   # pixel ruler in full-image coordinates (1920 x 1080), for choosing --crop
            f = img.shape[1] / 1920.0
            ax.set_xticks([x * f for x in range(0, 1921, 240)], [str(x) for x in range(0, 1921, 240)], fontsize=7)
            ax.set_yticks([y * f for y in range(0, 1081, 180)], [str(y) for y in range(0, 1081, 180)], fontsize=7)
            ax.grid(color="white", alpha=0.35, lw=0.6)
            if a.crop:        # preview of the crop rectangle
                x0, y0, x1, y1 = a.crop
                ax.add_patch(Rectangle((x0 * f, y0 * f), (x1 - x0) * f, (y1 - y0) * f,
                                       fill=False, ec="#FEC601", lw=2))
    show(axes.flat[5], micro_image(micro, t, (640, 483)), "no microscope frame")
    axes.flat[5].set_title("microscope", fontsize=12)
    fig.suptitle(f"Run {a.run} at t = {t:.1f} s ({PHASE_NAMES.get(phase_at(phases, t), '-')})", fontsize=13)
    fig.tight_layout()
    out = f"fig3_run{a.run:02d}_views_{t:.0f}.png"
    fig.savefig(out, dpi=80)
    print(f"Written: {out}")


def badge(ax, x, y, text, color, dark=False, size=7.5):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, fontweight="bold",
            color=INK if dark else "white", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.95))


def final(a, cams, micro, phases):
    n = len(a.times)
    letters = [f"({chr(97 + j)})" for j in range(n)]
    tags = (a.tags or []) + [""] * n
    ps.apply(7.5)
    room0 = room_image(cams, a.camera, a.times[0], a.crop)
    aspect_room = room0.shape[0] / room0.shape[1] if room0 is not None else 0.5625
    W = 7.16
    col_w = (W - 0.35) / n
    h_room, h_micro = col_w * aspect_room, col_w * 0.754
    h_strip, h_tag = 0.42, 0.22 if any(tags[:n]) else 0.05
    H = h_strip + 0.2 + h_room + 0.05 + h_micro + h_tag + 0.05
    fig = plt.figure(figsize=(W, H))
    left = 0.33 / W

    # phase strip with the moments
    t_lo, t_hi = min(a.times) - a.window, max(a.times) + a.window
    ax = fig.add_axes([left + 0.02 / W, 1 - (h_strip - 0.02) / H, 1 - left - 0.2 / W, (h_strip - 0.18) / H])
    for s, e, p in phases:
        s2, e2 = max(s, t_lo), min(e, t_hi)
        if e2 <= s2:
            continue
        ax.add_patch(Rectangle((s2, 0), e2 - s2, 1, color=PHASE_COLORS.get(p, "#999999"), lw=0))
        if e2 - s2 > 0.12 * (t_hi - t_lo):
            # put the phase name in the widest stretch between the (a)-(d) marker lines, so no line crosses it
            cuts = sorted([s2, e2] + [t for t in a.times if s2 < t < e2])
            g0, g1 = max(zip(cuts, cuts[1:]), key=lambda g: g[1] - g[0])
            ax.text((g0 + g1) / 2, 0.5, PHASE_NAMES.get(p, p), ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=INK if p in DARK_TEXT else "white")
    for (s, e, p), (s2, e2, p2) in zip(phases, phases[1:]):
        if t_lo < s2 < t_hi:
            ax.plot([s2, s2], [-0.15, 1.15], color=INK, lw=1, clip_on=False)
    for t, lt in zip(a.times, letters):
        ax.plot([t, t], [0, 1], color="white", lw=2.6, solid_capstyle="butt")
        ax.plot([t, t], [0, 1], color=INK, lw=1.0)
        ax.text(t, 1.08, lt, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=INK)
    ax.set_xlim(t_lo, t_hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=2, pad=1, labelsize=6.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f} s"))
    ax.text(1.0, 1.08, f"Run {a.run}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK)

    # image grid
    y_room = 1 - (h_strip + 0.2 + h_room) / H
    y_micro = y_room - (0.05 + h_micro) / H
    for j, t in enumerate(a.times):
        p = phase_at(phases, t)
        color = PHASE_COLORS.get(p, "#999999")
        x = left + j * col_w / W
        ax1 = fig.add_axes([x + 0.02 / W, y_room, (col_w - 0.04) / W, h_room / H])
        ax2 = fig.add_axes([x + 0.02 / W, y_micro, (col_w - 0.04) / W, h_micro / H])
        show(ax1, room_image(cams, a.camera, t, a.crop))
        show(ax2, micro_image(micro, t), "no microscope frame")
        for axx in (ax1, ax2):
            frame(axx, color)
        badge(ax1, 0.03, 0.96, f"{letters[j]}  {t:.0f} s", color, p in DARK_TEXT)
        if tags[j]:
            ax2.text(0.5, -0.04, tags[j], transform=ax2.transAxes, ha="center", va="top", fontsize=7.5,
                     fontweight="bold", color=INK)
        if j == 0:
            ax1.set_ylabel(f"Camera {int(a.camera[-1])}", fontsize=7.5, color=INK)
            ax2.set_ylabel("Microscope", fontsize=7.5, color=INK)
    out = f"fig3_run{a.run:02d}"
    fig.savefig(out + ".pdf")
    fig.savefig(out + ".png", dpi=220)
    print(f"Written: {out}.pdf / .png (upload the PDF to Overleaf as Figure_3_cross_scale_sequence.pdf)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--camera", default="camera03", choices=bv.CAMERAS)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sheet", type=float, nargs=2, metavar=("T0", "T1"))
    mode.add_argument("--views", type=float, metavar="T")
    mode.add_argument("--times", type=float, nargs="+")
    ap.add_argument("--tags", nargs="+", help="short label per time, e.g. \"Lights off\"")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=25.0)
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    a = ap.parse_args()
    if bv.cv2 is None:
        sys.exit("opencv-python is needed for the microscope video: pip install opencv-python")
    cams, micro, phases = setup(a)
    if a.sheet:
        sheet(a, cams, micro, phases)
    elif a.views is not None:
        views(a, cams, micro, phases)
    else:
        final(a, cams, micro, phases)


if __name__ == "__main__":
    main()
