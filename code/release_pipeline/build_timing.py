"""Bifocal-OR: timing layer of the release (sync/), from the collected sync files.

Writes into OUT/sync/:
  room_frame_times/runNN.csv            frame, t_s (seconds since the first frame), unix_ns
                                        from export_full/tables_timestamps.arrow (one time per frame,
                                        shared by the five cameras)
  microscope_frame_times/runNN[_partK].csv
                                        frame, t_s on the microscope clock (= the microscope's
                                        audio_feed.wav), from the ffmpeg encoder log (about every 0.5 s;
                                        interpolate linearly in between)
  offsets.csv                           where the audio and the microscope clock start on the
                                        room-camera timeline (offset_s; rate 1)
  README.md                             how to use the three

Method:
  room camera t = 0 ............ first exported frame (tables_timestamps.arrow, UTC ns)
  microscope clock 0 ........... "System Time at Start" in sync_log.txt (microscope PC, local time)
  synced audio t = 0 ........... found by cross-correlating the microscope microphone (audio_feed.wav)
                                 with the synced Room.wav (sync_audio_feeds.py -> audio_sync.csv)
  -> microscope offset_s = microscope start - first room frame (both PCs' clocks)
  -> audio offset_s      = microscope offset_s - (position of audio_feed on the synced timeline)
The step across the two PCs' clocks is good to about a second; the clap check refines it.

Usage:
  python build_timing.py SYNC_FILES_DIR AUDIO_SYNC_CSV OUT_DIR [--utc-offset-h 1] [--verified verified.csv]
    SYNC_FILES_DIR  unzipped sync_files.zip (collect_sync_files.py), .arrow.csv.gz tables
    OUT_DIR         the release metadata folder (Bifocal-OR)
    --verified      optional CSV run,audio_offset_s with clap-verified audio offsets; the same
                    correction is then applied to the microscope rows of that run
"""
import argparse
import csv
import datetime as dt
import glob
import gzip
import os
import re
import sys


def room_times(sync_dir, run):
    hits = glob.glob(os.path.join(sync_dir, f"rec_*_Run_{run}", "export_full", "tables_timestamps.arrow.csv.gz"))
    if not hits:
        return None
    with gzip.open(hits[0], "rt") as fh:
        r = csv.reader(fh)
        head = next(r)
        col = head.index("camera01_colorimage")
        rows = [(int(x[0]), int(x[col])) for x in r]
    return rows


def micro_anchors(log_path):
    text = open(log_path, encoding="utf-8", errors="replace").read()
    pts = []
    for line in re.split(r"[\r\n]", text):
        m = re.search(r"frame=\s*(\d+).*elapsed=(\d+):(\d+):([\d.]+)", line)
        if m:
            pts.append((int(m.group(1)), int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4))))
    clean = []
    for f, t in pts:  # keep strictly increasing frame numbers
        if not clean or f > clean[-1][0]:
            clean.append((f, t))
    return clean


def write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


README = """# Timing of the streams

All annotations and offsets refer to the **room-camera timeline**: t = 0 is the first exported
frame of the run, and every frame's time is in `room_frame_times/runNN.csv` (also in
`runs/runNN/room/timestamps.arrow`). The cameras were exported at a nominal 7.5 frames/s
(133.4 ms apart, {mean_fps} frames/s on average because single frames are occasionally missing),
so convert between frames and seconds with this table, not with a constant rate.

- **Phase labels** (`annotations/phases`) are in seconds on this timeline.
- **Synchronized audio** (`runs/runNN/audio/*.wav`) starts at `offset_s` of the `audio` row in
  `offsets.csv`: audio time = t - offset_s. Speaker labels (`annotations/speakers`) are in audio time.
- **Microscope:** the video files are stored at 29 frames/s, but the camera delivered fewer frames
  (about 21-29 per second, varying), so the stored video runs faster than real time. The real time of
  each frame is in `microscope_frame_times/runNN[_partK].csv` (frame, t_s on the microscope clock;
  interpolate linearly). The microscope clock starts at `offset_s` of the `microscope` row:
  microscope time = t - offset_s. The microphone recorded with it (`microscope_audio.wav`) runs on
  the same clock.

Status: {status}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sync_dir")
    ap.add_argument("audio_sync")
    ap.add_argument("out")
    ap.add_argument("--utc-offset-h", type=float, default=1.0, help="local time of the microscope PC minus UTC")
    ap.add_argument("--verified", help="CSV run,audio_offset_s from the clap check")
    ap.add_argument("--default-correction", type=float, default=0.0,
                    help="seconds added to runs without a clap check (e.g. the correction found on verified runs)")
    a = ap.parse_args()
    sync = os.path.join(a.out, "sync")
    feeds = {(int(r["run"]), r["part"]): float(r["offset_s"]) for r in csv.DictReader(open(a.audio_sync, encoding="utf-8"))}
    verified = {}
    if a.verified:
        verified = {int(r["run"]): float(r["audio_offset_s"]) for r in csv.DictReader(open(a.verified, encoding="utf-8"))}
    offsets, fps_all = [], []
    for run in range(1, 10):
        rows = room_times(a.sync_dir, run)
        if rows is None:
            print(f"Run {run}: no timestamps, skipped")
            continue
        t0 = rows[0][1]
        write(os.path.join(sync, "room_frame_times", f"run{run:02d}.csv"), ["frame", "t_s", "unix_ns"],
              [(f, f"{(ns - t0) / 1e9:.4f}", ns) for f, ns in rows])
        fps_all.append((len(rows) - 1) / ((rows[-1][1] - t0) / 1e9))
        cam0 = dt.datetime.fromtimestamp(t0 / 1e9, dt.timezone.utc).replace(tzinfo=None) + dt.timedelta(hours=a.utc_offset_h)
        logs = sorted(glob.glob(os.path.join(a.sync_dir, f"rec_*_Run_{run}", "*Manual*", "sync_log.txt")))
        parts = []
        for lg in logs:
            part = os.path.basename(os.path.dirname(lg))
            start = re.search(r"Start: (\S+ \S+)", open(lg, encoding="utf-8").read()).group(1)
            m_off = (dt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S.%f") - cam0).total_seconds()
            anchors = micro_anchors(os.path.join(os.path.dirname(lg), "ffmpeg_log.txt"))
            stream = "microscope" if len(logs) == 1 else f"microscope_part{len(parts) + 1}"
            name = f"run{run:02d}.csv" if len(logs) == 1 else f"run{run:02d}_part{len(parts) + 1}.csv"
            write(os.path.join(sync, "microscope_frame_times", name), ["frame", "t_s"],
                  [(f, f"{t:.2f}") for f, t in anchors])
            parts.append((stream, part, m_off, feeds.get((run, part)), anchors))
        audio_est = [m_off - fo for _, _, m_off, fo, _ in parts if fo is not None]
        a_off = audio_est[0] if audio_est else None
        corr = (verified[run] - a_off) if (run in verified and a_off is not None) else a.default_correction
        if run in verified:
            status = "clap-verified"
        elif a.default_correction:
            status = f"clocks + {a.default_correction:+.2f} s correction from the clap-verified runs (about +-0.5 s)"
        else:
            status = "estimated from clocks (+-1 s); verify with the clap check"
        note = "synced WAVs (audio/); t_audio = t - offset_s"
        if len(audio_est) > 1:
            note += f"; the two microscope parts give {audio_est[0]:.2f} and {audio_est[1]:.2f} s"
        offsets.append([f"run{run:02d}", "audio", f"{a_off + corr:.3f}" if a_off is not None else "", "1.0", status,
                        "microscope-PC start time + cross-correlation of the microscope microphone with Room.wav",
                        note])
        for stream, part, m_off, fo, anchors in parts:
            offsets.append([f"run{run:02d}", stream, f"{m_off + corr:.3f}", "1.0", status,
                            "sync_log.txt start time; frame times from the ffmpeg encoder log",
                            f"{part}; frame times in microscope_frame_times/; {anchors[-1][0]} frames over "
                            f"{anchors[-1][1]:.1f} s ({anchors[-1][0] / anchors[-1][1]:.1f} frames/s on average)"])
        print(f"Run {run}: audio offset {a_off + corr if a_off is not None else float('nan'):.3f} s; "
              + "; ".join(f"{s} {m + corr:+.3f} s" for s, _, m, _, _ in parts))
    write(os.path.join(sync, "offsets.csv"), ["run", "stream", "offset_s", "rate", "status", "method", "note"], offsets)
    st = "all offsets clap-verified" if len(verified) >= 9 else \
        "offsets estimated from the two recording PCs' clocks (about +-1 s) and being verified with the start claps"
    open(os.path.join(sync, "README.md"), "w", encoding="utf-8").write(
        README.format(mean_fps=f"{sum(fps_all) / len(fps_all):.3f}" if fps_all else "7.48", status=st))
    print(f"Written to {sync}")


if __name__ == "__main__":
    sys.exit(main())
