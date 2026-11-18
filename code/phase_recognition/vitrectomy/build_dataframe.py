"""
Builds the per-camera vitrectomy 1-fps dataframe for Stage 1 CNN feature
extraction -- timestamp-exact version (drop-in replacement, Nov 2026).

What changed against the original:
  * Labels are assigned with the real capture time of every frame
    (export_full/tables_timestamps.arrow, one UTC timestamp per frame,
    identical for all five cameras) instead of a constant rate estimated as
    n_frames / annotation duration. That estimate (~7.73 fps) was off because the
    exports run at 7.48 fps on average and the recording continues 50-80 s after
    the last annotated second; it placed phase boundaries up to 3.3 % of the
    elapsed time late (~35 s at the start of Microsurgery).
  * Sampling: the frame nearest to each integer second of capture time
    (true 1 Hz, as the original docstring intended). The original took every
    round(fps) = 8th frame, i.e. one sample per ~1.07 s. Use --sampling step8 to
    keep the old sampling with the new labels.

Output format, file names and columns are unchanged (image_path, video_idx,
class, frame_idx), so the rest of the pipeline runs as before.
Needs pyarrow:  pip install pyarrow

Usage
-----
    python vitrectomy/build_dataframe.py --camera camera01
    bash build_all_dataframes.sh
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── DEFAULT CONFIG ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "raw"
DEFAULT_ANN_ROOT = REPO_ROOT / "annotations"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "vitrectomy" / "dataframes"

# Format: (folder_name, video_idx, annotation_csv_filename)
RUNS = [
    ("rec_12-03-2026_11-25_Run_1", 0, "annotations_1.csv"),
    ("rec_12-03-2026_12-40_Run_2", 1, "annotations_2.csv"),
    ("rec_12-03-2026_14-02_Run_3", 2, "annotations_3.csv"),
    ("rec_12-03-2026_14-52_Run_4", 3, "annotations_4.csv"),
    ("rec_13-03-2026_09-32_Run_5", 4, "annotations_5.csv"),
    ("rec_13-03-2026_10-42_Run_6", 5, "annotations_6.csv"),
    ("rec_13-03-2026_13-26_Run_7", 6, "annotations_7.csv"),
    ("rec_13-03-2026_14-19_Run_8", 7, "annotations_8.csv"),
    ("rec_13-03-2026_15-37_Run_9", 8, "annotations_9.csv"),
]


# ── HELPERS ──────────────────────────────────────────────────────────────────
def frame_times(export_dir, camera):
    """Seconds since the first frame, indexed by frame number (NaN where a number is missing)."""
    import pyarrow.feather as feather
    col = f"{camera}_colorimage"
    t = feather.read_table(export_dir / "tables_timestamps.arrow", columns=["frame_number", col])
    fr = np.asarray(t.column(0).to_pylist(), dtype=np.int64)
    ns = np.asarray(t.column(1).to_pylist(), dtype=np.int64)
    out = np.full(fr.max() + 1, np.nan)
    out[fr] = (ns - ns.min()) / 1e9
    return out


def label_at(times, ann_df):
    """Phase id for every frame number (-1 before/after/between annotated segments)."""
    cls = np.full(len(times), -1, dtype=int)
    for row in ann_df.itertuples():
        cls[(times >= row.start_sec) & (times < row.end_sec)] = int(row.phase_id)
    last = ann_df.loc[ann_df["end_sec"].idxmax()]
    cls[times == last["end_sec"]] = int(last["phase_id"])   # include the final second
    return cls


def select_frames(frame_numbers, times, sampling):
    """Frame numbers to keep: nearest frame to each integer second, or every 8th frame."""
    if sampling == "step8":
        return frame_numbers[::8]
    t = times[frame_numbers]
    ok = ~np.isnan(t)
    fn, t = frame_numbers[ok], t[ok]
    order = np.argsort(t)
    fn, t = fn[order], t[order]
    grid = np.arange(0, int(t[-1]) + 1)
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    nearest = np.where(np.abs(t[idx - 1] - grid) <= np.abs(t[idx] - grid), idx - 1, idx)
    return np.unique(fn[nearest])


# ── MAIN ─────────────────────────────────────────────────────────────────────
def build(camera, dataset_root, ann_root, out_dir, sampling="second"):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vitrectomy_1fps_{camera}.pkl"
    rows = []

    for run_folder, vid_idx, ann_csv in RUNS:
        export_dir = dataset_root / run_folder / "export_full"
        color_dir = export_dir / "color"
        if not color_dir.exists():
            print(f"WARNING: {color_dir} does not exist, skipping run {vid_idx}")
            continue
        jpgs = {int(p.stem.split("_")[1]): p for p in color_dir.glob(f"color_*_{camera}.jpg")}
        if not jpgs:
            print(f"WARNING: no {camera} frames in {color_dir}, skipping run {vid_idx}")
            continue

        times = frame_times(export_dir, camera)
        ann_df = pd.read_csv(ann_root / ann_csv)
        classes = label_at(times, ann_df)
        frame_numbers = np.array(sorted(n for n in jpgs if n < len(times)))
        keep = select_frames(frame_numbers, times, sampling)

        dur = np.nanmax(times)
        print(f"\nRun {run_folder}  [{camera}]:")
        print(f"  {len(jpgs)} frames over {dur:.1f} s ({(len(times) - 1) / dur:.3f} fps from timestamps)")
        print(f"  Annotation: {ann_df['start_sec'].min():.0f}-{ann_df['end_sec'].max():.0f} s")
        print(f"  Selected {len(keep)} frames ({sampling}), {(classes[keep] >= 0).sum()} labeled")

        for n in keep:
            rows.append({
                "image_path": str(jpgs[n]),
                "video_idx": vid_idx,
                "class": int(classes[n]),
                "frame_idx": int(n),
            })

    if not rows:
        raise SystemExit(f"No frames collected for camera={camera}. "
                         f"Check {dataset_root} and {ann_root}.")

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 60}")
    print(f"Camera: {camera}   |   total frames: {len(df)}")
    print("Class distribution:")
    print(df["class"].value_counts().sort_index().to_string())
    labeled = (df["class"] >= 0).sum()
    unlabeled = (df["class"] == -1).sum()
    print(f"\nLabeled:   {labeled} ({100 * labeled / len(df):.1f}%)")
    print(f"Unlabeled: {unlabeled} ({100 * unlabeled / len(df):.1f}%)")

    df.to_pickle(out_path)
    print(f"\nSaved to {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="camera01",
                    help="Camera identifier (camera01..camera05). Default: camera01")
    ap.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT),
                    help="Path holding the rec_*/export_full/{color,tables_timestamps.arrow} trees")
    ap.add_argument("--ann-root", default=str(DEFAULT_ANN_ROOT),
                    help="Folder containing annotations_<n>.csv")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help="Where to write vitrectomy_1fps_<camera>.pkl")
    ap.add_argument("--sampling", choices=["second", "step8"], default="second",
                    help="second: frame nearest to each integer second (default); step8: every 8th frame")
    args = ap.parse_args()

    build(camera=args.camera,
          dataset_root=Path(args.dataset_root),
          ann_root=Path(args.ann_root),
          out_dir=Path(args.out_dir),
          sampling=args.sampling)


if __name__ == "__main__":
    main()
