#!/usr/bin/env python3
"""
measure_drift.py — Measure residual synchronization drift between each wearable
microphone and the ambient room microphone across a run.

For several windows spread across the run (start ... middle ... end), it
cross-correlates each wearable track against the room track and reports the
time offset (in milliseconds) that best aligns them. A perfectly synced track
reads ~0 ms everywhere. A linearly-drifting track shows offsets that grow
roughly linearly from start to end. A track that bows in the middle (offsets
near 0 at both ends but large in between) indicates non-linear drift that a
single linear stretch cannot fix — which is the case you observed.

Use it to (a) confirm which tracks drift and by how much, and (b) get the
measured worst-case number to put in the paper's synchronization paragraph.

Usage:
    python measure_drift.py --run-dir "G:\\Ophthalmo_4D_DVR_Sync\\Run_4"
    python measure_drift.py --run-dir "G:\\...\\Run_4" --room Run_4_Room.wav
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
from scipy.signal import correlate


def load_mono(path, target_sr):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    if sr != target_sr:
        # cheap linear resample; fine for envelope cross-correlation
        n_new = int(round(len(data) * target_sr / sr))
        data = np.interp(np.linspace(0, len(data), n_new, endpoint=False),
                         np.arange(len(data)), data).astype(np.float32)
    return data


def envelope(x, sr, env_hz=50):
    """Coarse amplitude envelope — robust for aligning speech energy."""
    x = np.abs(x)
    win = max(1, int(sr / env_hz))
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, kernel, mode="same")


def best_offset_ms(a, b, sr, max_lag_s=3.0):
    """Lag (ms) to add to `b` to align it with `a`, via envelope xcorr."""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    corr = correlate(a, b, mode="full")
    lags = np.arange(-len(b) + 1, len(a))
    max_lag = int(max_lag_s * sr)
    mid = len(corr) // 2
    lo, hi = mid - max_lag, mid + max_lag + 1
    seg = corr[lo:hi]
    seg_lags = lags[lo:hi]
    best = seg_lags[int(np.argmax(seg))]
    return 1000.0 * best / sr


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--run-dir", required=True)
    p.add_argument("--room", default=None,
                   help="room/ambient filename (default: the *Room*.wav in the folder)")
    p.add_argument("--sr", type=int, default=4000, help="analysis sample rate")
    p.add_argument("--windows", type=int, default=7, help="probe windows across the run")
    p.add_argument("--win-s", type=float, default=20.0, help="window length (s)")
    a = p.parse_args()

    wavs = [os.path.basename(x) for x in glob.glob(os.path.join(a.run_dir, "*.wav"))]
    room = a.room or next((w for w in wavs if "room" in w.lower()
                           or "ambient" in w.lower()), None)
    if room is None:
        print("No room/ambient WAV found; pass --room.", file=sys.stderr)
        return 2
    others = [w for w in wavs if w != room]
    print(f"Room reference : {room}")
    print(f"Wearables      : {others}\n")

    room_sig = load_mono(os.path.join(a.run_dir, room), a.sr)
    room_env = envelope(room_sig, a.sr)
    total_s = len(room_sig) / a.sr
    centers = np.linspace(0.12, 0.88, a.windows) * total_s  # avoid extreme edges

    header = "track".ljust(22) + "".join(f"{int(c):>7}s" for c in centers) + "   max|ms|"
    print(header)
    print("-" * len(header))
    worst_overall = 0.0
    for w in others:
        sig = load_mono(os.path.join(a.run_dir, w), a.sr)
        env = envelope(sig, a.sr)
        offs = []
        for c in centers:
            s = int((c - a.win_s / 2) * a.sr)
            e = int((c + a.win_s / 2) * a.sr)
            s, e = max(0, s), min(len(room_env), e, len(env))
            off = best_offset_ms(room_env[s:e], env[s:e], a.sr)
            offs.append(off)
        cells = "".join(f"{(f'{o:+.0f}' if o is not None else '  --'):>8}" for o in offs)
        valid = [abs(o) for o in offs if o is not None]
        worst = max(valid) if valid else 0.0
        worst_overall = max(worst_overall, worst)
        print(f"{w[:22]:22}{cells}   {worst:6.0f}")

    print("\nReading guide:")
    print("  ~0 everywhere            -> well synced.")
    print("  grows linearly start→end -> linear drift (one stretch could fix).")
    print("  ~0 at ends, large middle -> NON-LINEAR drift (a single stretch cannot fix).")
    print(f"\nWorst residual offset across all wearables: {worst_overall:.0f} ms")
    print("Use this number in the paper's synchronization paragraph (Section 3.3).")
    return 0


if __name__ == "__main__":
    sys.exit(main())