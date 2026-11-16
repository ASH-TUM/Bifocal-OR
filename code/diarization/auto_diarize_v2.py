#!/usr/bin/env python3
"""
auto_diarize_v2.py — Weak-supervision speaker diarization for the Micro-OR dataset.

Refined pipeline (Two-Pass VAD + Contrast Referee + Observability):

    1. Load every smartphone WAV in a run folder, resample to 16 kHz mono.
    2. Apply a zero-phase 300–3000 Hz bandpass ("comparison" copy) using sosfiltfilt.
    3. Silero VAD (Pass 1) on each RAW track, with neg_threshold hysteresis to
       suppress fragmentation in noisy OR audio.
    4. Robust per-mic gain calibration from the P90 RMS of *VAD-positive* bins
       (not from the peak — transients no longer ruin the referee).
    5. Contrast-based cross-talk resolution: at every 100 ms bin, each mic's
       calibrated RMS is compared against the median of all *other* mics'
       calibrated RMS. The winner is the VAD-active mic with the highest
       contrast, and must exceed MIN_CONTRAST_DB, otherwise the bin is UNKNOWN.
    6. Median-filter the integer-label timeline (Floor-Hold rule, kernel≈1.1 s).
    7. Emit labels, apply padding that is *clipped* against neighbouring-
       speaker blocks (never crosses another speaker).
    8. Merge same-speaker blocks within SAME_SPEAKER_MERGE_MS only if no other
       speaker is labelled in the gap.
    9. Write Audacity .txt labels + per-label confidence .csv + a per-run
       summary .json with mic-health diagnostics.

The output label file is bit-for-bit compatible with Audacity's Import Labels,
so this is a drop-in replacement for the prior script. The sidecar CSV and
summary are additions.

Usage:
    python auto_diarize_v2.py                                   # default config
    python auto_diarize_v2.py --base-dir G:\\Ophthalmo_4D_DVR_Sync
    python auto_diarize_v2.py --run Run_03                      # single run only
    python auto_diarize_v2.py --min-contrast-db 4.5 --no-csv    # tune + silence CSVs

Author: Angelo Henriques, TUM / Micro-OR Dataset project.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import butter, sosfiltfilt, medfilt

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    # I/O
    base_dir: Path = Path(r"G:\Ophthalmo_4D_DVR_Sync")
    run_filter: Optional[str] = None          # e.g. "Run_03" — process one run only
    exclude_patterns: tuple[str, ...] = ("Room", "Ambient", "_Master")
    label_suffix: str = "_AntiFragLabels.txt"
    write_confidence_csv: bool = True
    write_summary: bool = True

    # Audio
    target_sr: int = 16_000
    bin_ms: int = 100                         # timeline bin size

    # Bandpass (voice band)
    bandpass_low_hz: float = 300.0
    bandpass_high_hz: float = 3000.0
    bandpass_order: int = 4

    # Silero VAD
    vad_threshold: float = 0.35               # positive threshold (raised from 0.2)
    vad_neg_threshold: float = 0.20           # hysteresis — must fall below this to end
    min_speech_ms: int = 250
    min_pause_ms: int = 800
    speech_pad_ms: int = 60                   # Silero-internal segment padding

    # Calibration / referee
    calib_percentile: float = 90.0            # target: equalize P90 RMS of active bins
    calib_target_rms: float = 0.1             # equalization target level (arbitrary unit)
    min_active_seconds: float = 5.0           # flag/skip mics with less VAD than this
    gain_clip: float = 1.25                   # clamp per-mic gain to [1/clip, clip]; 1.0 = raw energy (no calibration)
    min_contrast_db: float = 2.0              # winner must beat median(others) by this
    smoothing_bins: int = 5                   # pre-referee rolling mean on RMS
    median_filter_window: int = 11            # Floor-Hold (≈ 1.1 s at 100 ms bins)

    # Label post-processing
    label_padding_ms: int = 200
    same_speaker_merge_ms: int = 1000

    # Runtime
    device: str = "auto"                      # "cpu", "cuda", or "auto"

    # Labels used internally
    unknown_label: str = "UNKNOWN"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


logger = logging.getLogger("auto_diarize")

# --------------------------------------------------------------------------- #
# Small utilities                                                             #
# --------------------------------------------------------------------------- #

def _build_bandpass(sample_rate: int, low: float, high: float, order: int):
    """Design a Butterworth bandpass and return SOS coefficients."""
    nyq = 0.5 * sample_rate
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _bandpass(data: np.ndarray, sos) -> np.ndarray:
    """Zero-phase bandpass using sosfiltfilt (better than tf-form filtfilt)."""
    return sosfiltfilt(sos, data).astype(np.float32, copy=False)


def _parse_role(filename: str) -> str:
    """Extract the role token from a WAV filename.

    Convention: Run_XX_<Role>.wav, sometimes <Role>_<misc>.wav. Falls back
    to the longest alphabetic token if the split-last heuristic is ambiguous.
    """
    stem = Path(filename).stem
    tail = stem.split("_")[-1]
    # If tail is purely numeric (e.g. Surgeon_Mic_1), pick the last alpha token.
    if tail.isdigit():
        alpha_tokens = [t for t in stem.split("_") if t.isalpha()]
        if alpha_tokens:
            return alpha_tokens[-1]
    # Strip trailing digits (Surgeon1 → Surgeon).
    return re.sub(r"\d+$", "", tail) or tail


def _load_wav_mono(path: Path, target_sr: int) -> torch.Tensor:
    """Load a WAV, downmix to mono if needed, resample to target_sr, return float32 tensor."""
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    tensor = torch.from_numpy(data)
    if sr != target_sr:
        tensor = torchaudio.functional.resample(tensor, orig_freq=sr, new_freq=target_sr)
    return tensor


# --------------------------------------------------------------------------- #
# Silero VAD loader                                                           #
# --------------------------------------------------------------------------- #

def load_silero(device: torch.device):
    logger.info("Loading Silero VAD model (device=%s)…", device)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    model = model.to(device)
    model.eval()
    return model, get_speech_timestamps


# --------------------------------------------------------------------------- #
# Per-channel analysis                                                        #
# --------------------------------------------------------------------------- #

@dataclass
class ChannelStats:
    role: str
    path: Path
    num_samples: int
    num_vad_segments: int = 0
    active_seconds: float = 0.0
    p90_active_rms: float = 0.0
    gain: float = 1.0
    healthy: bool = True
    note: str = ""


def run_vad_and_rms(
    raw: torch.Tensor,
    filtered: np.ndarray,
    model,
    get_speech_timestamps,
    cfg: Config,
    device: torch.device,
    role: str,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, ChannelStats]:
    """Returns:
        vad_mask_bins   : np.ndarray[bool]  — True where VAD says speech
        rms_bins        : np.ndarray[float] — raw bandpassed RMS per bin
        stats           : ChannelStats      — per-channel diagnostics
    """
    bin_size = int(cfg.bin_ms * 1e-3 * cfg.target_sr)
    n_bins = (raw.numel() + bin_size - 1) // bin_size
    stats = ChannelStats(role=role, path=path, num_samples=raw.numel())

    # --- Silero VAD (Pass 1) -------------------------------------------------
    with torch.inference_mode():
        segments = get_speech_timestamps(
            raw.to(device),
            model,
            sampling_rate=cfg.target_sr,
            threshold=cfg.vad_threshold,
            neg_threshold=cfg.vad_neg_threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_pause_ms,
            speech_pad_ms=cfg.speech_pad_ms,
        )

    vad_mask = np.zeros(n_bins, dtype=bool)
    for seg in segments:
        s = seg["start"] // bin_size
        e = min(n_bins, (seg["end"] + bin_size - 1) // bin_size)
        vad_mask[s:e] = True

    active_seconds = float(vad_mask.sum()) * cfg.bin_ms / 1000.0
    stats.num_vad_segments = len(segments)
    stats.active_seconds = active_seconds

    # --- Global per-bin RMS on the BANDPASSED copy ---------------------------
    # Pad filtered signal up to n_bins * bin_size, then reshape and RMS.
    pad_needed = n_bins * bin_size - len(filtered)
    if pad_needed > 0:
        filtered = np.pad(filtered, (0, pad_needed), mode="constant")
    rms_bins = np.sqrt(
        np.mean(filtered[: n_bins * bin_size].reshape(n_bins, bin_size) ** 2, axis=1)
    ).astype(np.float32)

    # --- Robust per-mic calibration ------------------------------------------
    # Use the P90 RMS from VAD-positive bins as the gain reference.
    if vad_mask.any():
        active_rms = rms_bins[vad_mask]
        p90 = float(np.percentile(active_rms, cfg.calib_percentile))
    else:
        p90 = 0.0
    stats.p90_active_rms = p90

    if active_seconds < cfg.min_active_seconds or p90 < 1e-6:
        stats.healthy = False
        stats.note = (
            f"dead/silent mic: {active_seconds:.1f}s active < "
            f"{cfg.min_active_seconds:.1f}s, P90 RMS={p90:.4g}"
        )
        stats.gain = 0.0                  # will be excluded from referee
    else:
        raw_gain = cfg.calib_target_rms / p90
        _hi = max(cfg.gain_clip, 1.0)
        stats.gain = float(np.clip(raw_gain, 1.0 / _hi, _hi))

    return vad_mask, rms_bins, stats


# --------------------------------------------------------------------------- #
# Cross-talk referee                                                          #
# --------------------------------------------------------------------------- #

def resolve_crosstalk(
    roles: list[str],
    vad_masks: dict[str, np.ndarray],
    rms_bins: dict[str, np.ndarray],
    gains: dict[str, float],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a winner per bin using contrast-based resolution.

    Returns:
        label_ints        : np.ndarray[int]   — 0 = silence/unknown, 1..K = role idx
        contrast_db_bins  : np.ndarray[float] — per-bin contrast (dB) of the winner
    """
    n_bins = len(next(iter(rms_bins.values())))
    k = len(roles)

    # Stack: (K, n_bins) calibrated RMS, zero out unhealthy mics.
    calibrated = np.zeros((k, n_bins), dtype=np.float32)
    vad_stack = np.zeros((k, n_bins), dtype=bool)
    for i, role in enumerate(roles):
        calibrated[i] = rms_bins[role] * gains[role]
        vad_stack[i] = vad_masks[role] & (gains[role] > 0.0)

    # Smooth calibrated RMS (rolling mean) to tame short spikes.
    if cfg.smoothing_bins > 1:
        kernel = np.ones(cfg.smoothing_bins, dtype=np.float32) / cfg.smoothing_bins
        for i in range(k):
            calibrated[i] = np.convolve(calibrated[i], kernel, mode="same")

    # For each mic candidate, contrast = own_rms / median(other mics' rms).
    eps = 1e-8
    contrast_db = np.full((k, n_bins), -np.inf, dtype=np.float32)

    # Vectorised: for each mic i, denominator = median over all other rows.
    # Use masked median via np.sort; k is small (typically 4–6) so this is cheap.
    if k <= 1:
        # Degenerate case — one mic only. Any VAD-positive bin wins trivially.
        contrast_db[0] = np.where(vad_stack[0], 60.0, -np.inf)
    else:
        for i in range(k):
            others = np.delete(calibrated, i, axis=0)
            denom = np.median(others, axis=0) + eps
            ratio = (calibrated[i] + eps) / denom
            contrast_db[i] = 20.0 * np.log10(ratio + eps)

    # Non-VAD-active mics are ineligible.
    contrast_db_masked = np.where(vad_stack, contrast_db, -np.inf)

    # Winner = argmax; require above threshold AND at least one VAD-active mic.
    winner_idx = np.argmax(contrast_db_masked, axis=0)  # 0..k-1
    winner_db = contrast_db_masked[winner_idx, np.arange(n_bins)]
    any_vad = vad_stack.any(axis=0)
    accepted = any_vad & (winner_db >= cfg.min_contrast_db)

    # label 0 = silence/unknown, 1..k = role index + 1.
    label_ints = np.where(accepted, winner_idx + 1, 0).astype(np.int32)

    # Floor-Hold median filter on labels.
    if cfg.median_filter_window > 1:
        label_ints = medfilt(label_ints, kernel_size=cfg.median_filter_window)

    # Return winner_db aligned with label_ints (replace rejected with NaN for csv).
    contrast_db_out = np.where(label_ints > 0, winner_db, np.nan).astype(np.float32)
    return label_ints, contrast_db_out


# --------------------------------------------------------------------------- #
# Label extraction, padding, merging                                          #
# --------------------------------------------------------------------------- #

@dataclass
class Label:
    start: float
    end: float
    role: str
    mean_contrast_db: float = 0.0
    padded_ms: int = 0
    merged_count: int = 1


def labels_from_timeline(
    label_ints: np.ndarray,
    contrast_db: np.ndarray,
    roles: list[str],
    cfg: Config,
) -> list[Label]:
    """Collapse the per-bin integer timeline to contiguous labelled spans."""
    bin_s = cfg.bin_ms / 1000.0
    out: list[Label] = []
    current = int(label_ints[0])
    start = 0
    contrast_acc: list[float] = []
    for b in range(1, len(label_ints)):
        v = int(label_ints[b])
        if not np.isnan(contrast_db[b - 1]):
            contrast_acc.append(float(contrast_db[b - 1]))
        if v != current:
            if current > 0:
                role = roles[current - 1]
                out.append(
                    Label(
                        start=start * bin_s,
                        end=b * bin_s,
                        role=role,
                        mean_contrast_db=float(np.mean(contrast_acc)) if contrast_acc else 0.0,
                    )
                )
            current = v
            start = b
            contrast_acc = []
    # flush tail
    if current > 0:
        role = roles[current - 1]
        if not np.isnan(contrast_db[-1]):
            contrast_acc.append(float(contrast_db[-1]))
        out.append(
            Label(
                start=start * bin_s,
                end=len(label_ints) * bin_s,
                role=role,
                mean_contrast_db=float(np.mean(contrast_acc)) if contrast_acc else 0.0,
            )
        )
    return out


def apply_clipped_padding(labels: list[Label], cfg: Config) -> list[Label]:
    """Pad each label by ±label_padding_ms, but clip against neighbours
    of a *different* speaker so we never cross another speaker's territory."""
    pad = cfg.label_padding_ms / 1000.0
    n = len(labels)
    for i, lab in enumerate(labels):
        orig_start, orig_end = lab.start, lab.end

        # left pad
        left_bound = 0.0
        for j in range(i - 1, -1, -1):
            if labels[j].role != lab.role:
                left_bound = labels[j].end
                break
        lab.start = max(left_bound, orig_start - pad)

        # right pad
        right_bound = float("inf")
        for j in range(i + 1, n):
            if labels[j].role != lab.role:
                right_bound = labels[j].start
                break
        lab.end = min(right_bound, orig_end + pad)

        lab.padded_ms = int(round((orig_start - lab.start + lab.end - orig_end) * 1000))
    return labels


def merge_same_speaker(labels: list[Label], cfg: Config) -> list[Label]:
    """Fuse same-speaker labels within SAME_SPEAKER_MERGE_MS ONLY IF no other
    speaker's label lies in the gap."""
    if not labels:
        return labels
    gap_max = cfg.same_speaker_merge_ms / 1000.0
    merged: list[Label] = [labels[0]]
    # We iterate in chronological order, so "previous same-speaker" is the
    # most recent one in `merged` with matching role that has no intervening
    # other-speaker label. The simplest safe check: look at merged[-1] only.
    for lab in labels[1:]:
        prev = merged[-1]
        can_merge = (
            lab.role == prev.role
            and (lab.start - prev.end) <= gap_max
        )
        if can_merge:
            # Weighted mean contrast.
            w_prev = max(prev.end - prev.start, 1e-6)
            w_cur = max(lab.end - lab.start, 1e-6)
            prev.mean_contrast_db = (
                prev.mean_contrast_db * w_prev + lab.mean_contrast_db * w_cur
            ) / (w_prev + w_cur)
            prev.end = max(prev.end, lab.end)
            prev.merged_count += lab.merged_count
            prev.padded_ms = max(prev.padded_ms, lab.padded_ms)
        else:
            merged.append(lab)
    return merged


# --------------------------------------------------------------------------- #
# Outputs                                                                     #
# --------------------------------------------------------------------------- #

def write_audacity_labels(labels: list[Label], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(f"{lab.start:.3f}\t{lab.end:.3f}\t{lab.role}\n")


def write_confidence_csv(labels: list[Label], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["start_s", "end_s", "duration_s", "role",
             "mean_contrast_db", "merged_count", "padded_ms"]
        )
        for lab in labels:
            w.writerow(
                [f"{lab.start:.3f}", f"{lab.end:.3f}",
                 f"{lab.end - lab.start:.3f}", lab.role,
                 f"{lab.mean_contrast_db:.2f}", lab.merged_count, lab.padded_ms]
            )


def write_run_summary(
    labels: list[Label],
    channel_stats: list[ChannelStats],
    total_duration_s: float,
    cfg: Config,
    path: Path,
) -> None:
    per_speaker: dict[str, dict[str, float]] = {}
    for lab in labels:
        d = lab.end - lab.start
        rec = per_speaker.setdefault(
            lab.role, {"labels": 0, "seconds": 0.0, "mean_contrast_db": 0.0}
        )
        rec["labels"] += 1
        rec["seconds"] += d
        rec["mean_contrast_db"] += lab.mean_contrast_db * d
    for rec in per_speaker.values():
        if rec["seconds"] > 0:
            rec["mean_contrast_db"] /= rec["seconds"]
        rec["percent_of_run"] = 100.0 * rec["seconds"] / total_duration_s

    summary = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(cfg).items()
        },
        "total_duration_s": round(total_duration_s, 3),
        "label_count": len(labels),
        "per_speaker": per_speaker,
        "mic_health": [
            {
                "role": s.role,
                "path": str(s.path),
                "healthy": s.healthy,
                "active_seconds": round(s.active_seconds, 2),
                "p90_active_rms": round(s.p90_active_rms, 5),
                "gain": round(s.gain, 4),
                "vad_segments": s.num_vad_segments,
                "note": s.note,
            }
            for s in channel_stats
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------------------------------- #
# Main per-run driver                                                         #
# --------------------------------------------------------------------------- #

def should_include(filename: str, exclude_patterns: tuple[str, ...]) -> bool:
    name = os.path.basename(filename)
    return not any(p in name for p in exclude_patterns)


def process_run(run_folder: Path, model, get_speech_timestamps, cfg: Config,
                device: torch.device) -> None:
    run_name = run_folder.name
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Processing run: %s", run_name)
    logger.info("=" * 60)

    wav_files = sorted(
        Path(p) for p in glob.glob(str(run_folder / "*.wav"))
        if should_include(p, cfg.exclude_patterns)
    )
    if not wav_files:
        logger.warning("No wav files in %s after exclusion — skipping.", run_folder)
        return

    sos = _build_bandpass(
        cfg.target_sr, cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order
    )

    roles: list[str] = []
    role_to_path: dict[str, Path] = {}
    raw_tracks: dict[str, torch.Tensor] = {}
    filtered_tracks: dict[str, np.ndarray] = {}

    logger.info("1/4  Loading + bandpassing %d tracks…", len(wav_files))
    for p in wav_files:
        role = _parse_role(p.name)
        if role in roles:
            logger.warning("Duplicate role %r from %s — appending suffix", role, p.name)
            role = f"{role}_{p.stem}"
        roles.append(role)
        role_to_path[role] = p

        raw = _load_wav_mono(p, cfg.target_sr)
        raw_tracks[role] = raw
        filtered_tracks[role] = _bandpass(raw.numpy(), sos)
        logger.info("     - %-20s  %7.1f s", role, raw.numel() / cfg.target_sr)

    # Align all tracks to the same length (they *should* already be; sync
    # pipeline exports uniform WAVs — but belt & braces).
    n_samples = min(t.numel() for t in raw_tracks.values())
    for role in roles:
        raw_tracks[role] = raw_tracks[role][:n_samples]
        filtered_tracks[role] = filtered_tracks[role][:n_samples]
    total_duration_s = n_samples / cfg.target_sr

    logger.info("2/4  Silero VAD + per-mic calibration…")
    vad_masks: dict[str, np.ndarray] = {}
    rms_bins: dict[str, np.ndarray] = {}
    gains: dict[str, float] = {}
    channel_stats: list[ChannelStats] = []

    for role in roles:
        vad_mask, rms, stats = run_vad_and_rms(
            raw=raw_tracks[role],
            filtered=filtered_tracks[role],
            model=model,
            get_speech_timestamps=get_speech_timestamps,
            cfg=cfg,
            device=device,
            role=role,
            path=role_to_path[role],
        )
        vad_masks[role] = vad_mask
        rms_bins[role] = rms
        gains[role] = stats.gain
        channel_stats.append(stats)
        flag = "OK" if stats.healthy else "DEAD"
        logger.info(
            "     - %-20s  [%s]  %4d segments  %6.1f s active  gain=%.3f",
            role, flag, stats.num_vad_segments, stats.active_seconds, stats.gain,
        )

    healthy_roles = [s.role for s in channel_stats if s.healthy]
    if not healthy_roles:
        logger.error("No healthy mics in %s — skipping.", run_name)
        return

    logger.info("3/4  Contrast referee + Floor-Hold median filter…")
    label_ints, contrast_db = resolve_crosstalk(
        roles=roles,
        vad_masks=vad_masks,
        rms_bins=rms_bins,
        gains=gains,
        cfg=cfg,
    )

    logger.info("4/4  Labels → padding → same-speaker merge → write…")
    labels = labels_from_timeline(label_ints, contrast_db, roles, cfg)
    labels = apply_clipped_padding(labels, cfg)
    labels = merge_same_speaker(labels, cfg)

    # Write outputs.
    label_path = run_folder / f"{run_name}{cfg.label_suffix}"
    write_audacity_labels(labels, label_path)
    logger.info("     → %s (%d labels)", label_path.name, len(labels))

    if cfg.write_confidence_csv:
        csv_path = run_folder / f"{run_name}{cfg.label_suffix.replace('.txt', '.csv')}"
        write_confidence_csv(labels, csv_path)
        logger.info("     → %s", csv_path.name)

    if cfg.write_summary:
        sum_path = run_folder / f"{run_name}_diarize_summary.json"
        write_run_summary(labels, channel_stats, total_duration_s, cfg, sum_path)
        logger.info("     → %s", sum_path.name)

    logger.info("Done %s in %.1f s.", run_name, time.time() - t0)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def _parse_args(argv: list[str]) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-dir", type=Path, default=cfg.base_dir)
    p.add_argument("--run", type=str, default=cfg.run_filter,
                   help="process only a single run folder name (e.g. Run_03)")
    p.add_argument("--vad-threshold", type=float, default=cfg.vad_threshold)
    p.add_argument("--vad-neg-threshold", type=float, default=cfg.vad_neg_threshold)
    p.add_argument("--min-contrast-db", type=float, default=cfg.min_contrast_db)
    p.add_argument("--gain-clip", type=float, default=cfg.gain_clip,
                   help="clamp per-mic gain to [1/clip, clip]; 1.0 = raw energy")
    p.add_argument("--median-window", type=int, default=cfg.median_filter_window)
    p.add_argument("--padding-ms", type=int, default=cfg.label_padding_ms)
    p.add_argument("--merge-ms", type=int, default=cfg.same_speaker_merge_ms)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default=cfg.device)
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-summary", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    cfg.base_dir = a.base_dir
    cfg.run_filter = a.run
    cfg.vad_threshold = a.vad_threshold
    cfg.vad_neg_threshold = a.vad_neg_threshold
    cfg.min_contrast_db = a.min_contrast_db
    cfg.gain_clip = a.gain_clip
    cfg.median_filter_window = a.median_window
    cfg.label_padding_ms = a.padding_ms
    cfg.same_speaker_merge_ms = a.merge_ms
    cfg.device = a.device
    cfg.write_confidence_csv = not a.no_csv
    cfg.write_summary = not a.no_summary

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])

    if not cfg.base_dir.is_dir():
        logger.error("Base dir not found: %s", cfg.base_dir)
        return 2

    device = cfg.resolve_device()
    model, get_speech_timestamps = load_silero(device)

    run_folders = sorted(
        Path(f.path) for f in os.scandir(cfg.base_dir)
        if f.is_dir() and "Run_" in f.name
    )
    if cfg.run_filter:
        run_folders = [rf for rf in run_folders if rf.name == cfg.run_filter]

    if not run_folders:
        logger.error("No Run_* folders found under %s", cfg.base_dir)
        return 1

    for folder in run_folders:
        try:
            process_run(folder, model, get_speech_timestamps, cfg, device)
        except Exception:
            logger.exception("Run %s failed — continuing.", folder.name)

    logger.info("All runs processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
