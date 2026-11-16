"""Bifocal-OR: place each microscope recording on the synced-audio timeline (read-only).

The microscope PC recorded a microphone next to the video (audio_feed.wav in each
Run_<n>_Manual folder). This script finds where that recording sits relative to the
synced room channel (Ophthalmo_4D_DVR_Sync/Run_<n>/Run_<n>_Room.wav) by cross-correlating
the two, first coarsely over the whole recording, then precisely at the start, middle and
end. The middle and end values show whether the two clocks drift apart.

It also checks the synchronized wearable tracks of each run (Run_<n>_<Role>.wav) against the
room track at five points (12, 31, 50, 69 and 88 % of the run). Perfectly synced tracks read
about 0 ms everywhere; the largest value is the residual offset to quote in the paper.

Usage (Windows PowerShell):
    python sync_audio_feeds.py "E:\\Ophthalmo_4D"
    python sync_audio_feeds.py "E:\\Ophthalmo_4D" --runs 3 4 --out audio_sync.csv
    python sync_audio_feeds.py "E:\\Ophthalmo_4D" --no-wearables

Output (CSV, one row per microscope recording):
    run, part, feed, offset_s   time on the synced-audio timeline at which the feed starts
    lag_start_s, lag_mid_s, lag_end_s   fine offsets measured at three points
    drift_ms                    lag_end - lag_start, in milliseconds
    corr                        correlation at the best lag in the fine windows (higher is better).
                                The match is reliable when start, middle and end agree within a few ms
                                (or drift smoothly)

Second output (wearable_sync.csv, one row per synced track):
    run, track, delay_12_ms ... delay_88_ms   delay of the track relative to the room track
                                (positive: the track is later), 10-ms envelope, interpolated
    max_abs_ms                  largest |delay| over the windows with corr >= 0.3
    min_corr, n_reliable        envelope correlation; windows below 0.3 (silent or dead
                                microphone) are reported but not counted

Reads about 2-4 GB per run from the drive; expect 2-3 minutes per run. Needs numpy.
"""
import argparse
import csv
import glob
import os
import re
import sys
import time
import wave

import numpy as np

ENV_RATE = 100          # envelope samples per second for the coarse search
FINE_RATE = 8000        # sample rate for the fine search
WIN = 30.0              # seconds per fine window
WEAR_POINTS = (0.12, 0.31, 0.50, 0.69, 0.88)   # where the wearable tracks are checked
WEAR_WIN = 60.0         # seconds per wearable window
WEAR_MIN_CORR = 0.3     # below this, a window is reported but not counted


def read_mono(path, start_s=0.0, dur_s=None, target_rate=None, chunk_s=20.0):
    """Mono float32 samples of a PCM WAV, optionally averaged down to target_rate."""
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported (found {8 * width}-bit)")
        a = max(0, int(start_s * sr))
        b = n if dur_s is None else min(n, a + int(dur_s * sr))
        w.setpos(a)
        step = max(1, int(round(sr / target_rate))) if target_rate else 1
        out, left = [], b - a
        per = int(chunk_s * sr) // step * step
        while left > 0:
            k = min(per, left)
            x = np.frombuffer(w.readframes(k), "<i2").astype(np.float32).reshape(-1, ch).mean(1) / 32768
            if step > 1:
                m = len(x) // step * step
                x = x[:m].reshape(-1, step).mean(1)
            out.append(x)
            left -= k
        return np.concatenate(out) if out else np.zeros(0, np.float32), sr / step, n / sr


def envelope(path):
    """Log energy per 10 ms over the whole file."""
    with wave.open(path, "rb") as w:
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported")
        hop = sr // ENV_RATE
        env = []
        while True:
            raw = w.readframes(hop * 2000)
            if not raw:
                break
            x = np.frombuffer(raw, "<i2").astype(np.float32).reshape(-1, ch).mean(1) / 32768
            m = len(x) // hop * hop
            if m:
                env.append(np.sqrt((x[:m].reshape(-1, hop) ** 2).mean(1)))
    e = np.log1p(1000 * np.concatenate(env)) if env else np.zeros(1)
    return (e - e.mean()) / (e.std() + 1e-9), n / sr


def xcorr_lag(a, b, max_lag=None):
    """Lag (in samples) that best aligns b to a: a[i] ~ b[i - lag]. Returns lag and quality."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    fa, fb = np.fft.rfft(a, n), np.fft.rfft(b, n)
    c = np.fft.irfft(fa * np.conj(fb), n)
    lags = np.arange(n)
    lags[lags > n // 2] -= n
    if max_lag is not None:
        keep = np.abs(lags) <= max_lag
        c, lags = c[keep], lags[keep]
    i = int(np.argmax(c))
    return int(lags[i]), float(c[i])


def xcorr_lag_interp(a, b, max_lag):
    """Like xcorr_lag, but with a sub-sample (parabolic) peak; returns lag (float) and correlation."""
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    c = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    c = np.concatenate([c[-max_lag:], c[:max_lag + 1]])      # lags -max_lag .. +max_lag
    i = int(np.argmax(c))
    d = 0.0
    if 0 < i < len(c) - 1:
        den = c[i - 1] - 2 * c[i] + c[i + 1]
        if den < 0:
            d = 0.5 * (c[i - 1] - c[i + 1]) / den
    return i - max_lag + d, float(c[i]) / len(a)


def wearable_delays(er, ew):
    """Delay (ms) of a synced track relative to the room track at WEAR_POINTS, from the envelopes."""
    n = min(len(er), len(ew))
    half = int(WEAR_WIN * ENV_RATE / 2)
    out = []
    for p in WEAR_POINTS:
        c = int(p * n)
        a, b = er[max(0, c - half):c + half], ew[max(0, c - half):c + half]
        if len(a) < 10 * ENV_RATE or b.std() < 1e-6:
            out.append((None, 0.0))
            continue
        a = (a - a.mean()) / (a.std() + 1e-9)
        b = (b - b.mean()) / (b.std() + 1e-9)
        lag, corr = xcorr_lag_interp(a, b, 3 * ENV_RATE)
        out.append((-lag * 1000.0 / ENV_RATE, corr))    # a[i] ~ b[i - lag]: the track is -lag late
    return out


def fine_lag(ref, feed, coarse_s, at_ref_s):
    """Precise lag around coarse_s, measured on a WIN-second window of the reference at at_ref_s."""
    r, sr, _ = read_mono(ref, at_ref_s, WIN, FINE_RATE)
    f, _, _ = read_mono(feed, at_ref_s - coarse_s - 1.0, WIN + 2.0, FINE_RATE)
    if len(r) < sr * 5 or len(f) < sr * 5:
        return None, 0.0
    r = (r - r.mean()) / (r.std() + 1e-9)
    f = (f - f.mean()) / (f.std() + 1e-9)
    lag, peak = xcorr_lag(r, f, max_lag=int(1.5 * sr))
    # r[i] ~ f[i - lag]  ->  ref time at_ref_s + i/sr = feed time (at_ref_s - coarse - 1) + (i - lag)/sr + offset
    return coarse_s + 1.0 + lag / sr, peak / len(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="clean dataset folder, e.g. E:\\Ophthalmo_4D")
    ap.add_argument("--runs", type=int, nargs="*", default=list(range(1, 10)))
    ap.add_argument("--out", default="audio_sync.csv")
    ap.add_argument("--wearables-out", default="wearable_sync.csv")
    ap.add_argument("--no-wearables", action="store_true", help="skip the check of the synced wearable tracks")
    args = ap.parse_args()
    # results are written after every run, so a stop or crash keeps the runs done so far
    fh = open(args.out, "w", newline="", encoding="utf-8")
    out = csv.writer(fh)
    out.writerow(["run", "part", "feed", "offset_s", "coarse_s", "lag_start_s", "lag_mid_s", "lag_end_s",
                  "drift_ms", "corr", "room_wav_s", "feed_s"])
    wfh = None if args.no_wearables else open(args.wearables_out, "w", newline="", encoding="utf-8")
    if wfh:
        wout = csv.writer(wfh)
        wout.writerow(["run", "track", *[f"delay_{int(100 * p)}_ms" for p in WEAR_POINTS],
                       "max_abs_ms", "min_corr", "n_reliable"])
    for run in args.runs:
        rows, wrows = [], []
        ref = os.path.join(args.root, "Ophthalmo_4D_DVR_Sync", f"Run_{run}", f"Run_{run}_Room.wav")
        if not os.path.exists(ref):
            print(f"Run {run}: {ref} not found, skipped")
            continue
        run_dirs = [d for d in glob.glob(os.path.join(args.root, "rec_*")) if re.search(rf"_Run_{run}$", d)]
        feeds = sorted(glob.glob(os.path.join(run_dirs[0], "*Manual*", "audio_feed.wav"))) if run_dirs else []
        t0 = time.time()
        er, dur_ref = envelope(ref)
        if not feeds:
            print(f"Run {run}: no microscope audio (audio_feed.wav) found")
        for feed in feeds:
            part = os.path.basename(os.path.dirname(feed))
            ef, dur_feed = envelope(feed)
            lag, _ = xcorr_lag(er, ef)
            coarse = lag / ENV_RATE          # feed t=0 sits at ref time 'coarse'
            lo, hi = max(coarse, 0.0), min(coarse + dur_feed, dur_ref)
            points = [lo + 20, (lo + hi) / 2 - WIN / 2, hi - WIN - 20] if hi - lo > 3 * WIN + 40 else [lo + 2]
            res = [fine_lag(ref, feed, coarse, p) for p in points]
            fines = [f for f, _ in res if f is not None]
            q = float(np.mean([c for f, c in res if f is not None])) if fines else 0.0
            offset = fines[0] if fines else coarse
            cells = [f"{f:.4f}" for f in fines] + [""] * (3 - len(fines))
            drift = f"{1000 * (fines[-1] - fines[0]):.1f}" if len(fines) > 1 else ""
            rows.append([run, part, os.path.relpath(feed, args.root), f"{offset:.4f}", f"{coarse:.2f}",
                         *cells, drift, f"{q:.2f}", f"{dur_ref:.2f}", f"{dur_feed:.2f}"])
            print(f"Run {run} {part}: feed starts at {offset:+.3f} s on the synced-audio timeline "
                  f"(coarse {coarse:+.2f} s, drift {drift or '-'} ms, corr {q:.2f}, {time.time() - t0:.0f} s)")
        out.writerows(rows)
        fh.flush()
        if args.no_wearables:
            continue
        worst = []
        for wav in sorted(glob.glob(os.path.join(os.path.dirname(ref), f"Run_{run}_*.wav"))):
            if os.path.basename(wav) == os.path.basename(ref):
                continue
            try:
                ew, dur_w = envelope(wav)
            except (ValueError, wave.Error, EOFError) as exc:
                print(f"Run {run}: {os.path.basename(wav)} skipped ({exc})")
                continue
            if abs(dur_w - dur_ref) > 1.0:      # short excerpts, e.g. *_for_video.wav
                continue
            res = wearable_delays(er, ew)
            good = [abs(d) for d, c in res if d is not None and c >= WEAR_MIN_CORR]
            track = os.path.basename(wav)[len(f"Run_{run}_"):-4]
            wrows.append([run, track, *[("" if d is None else f"{d:.1f}") for d, _ in res],
                          f"{max(good):.1f}" if good else "", f"{min(c for _, c in res):.2f}", len(good)])
            if good:
                worst.append((max(good), track))
        if worst:
            m, track = max(worst)
            print(f"Run {run} synced tracks: largest residual offset {m:.0f} ms ({track}); "
                  f"{len(worst)} tracks checked, {time.time() - t0:.0f} s")
        wout.writerows(wrows)
        wfh.flush()
    fh.close()
    print(f"\nWritten: {os.path.abspath(args.out)}")
    if wfh:
        wfh.close()
        print(f"Written: {os.path.abspath(args.wearables_out)}")


if __name__ == "__main__":
    main()
