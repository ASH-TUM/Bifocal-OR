#!/usr/bin/env python3
"""
ambient_diarize.py — Far-field speaker diarization on the ambient room mic.

Runs pyannote.audio's pretrained pipeline on a single ambient WAV and writes
the result as an Audacity label file (start <TAB> end <TAB> SPEAKER_xx) that
score_diarization.py can read directly.

Key trick: we load the WAV ourselves with soundfile and hand pyannote an
in-memory {"waveform", "sample_rate"} dict. This avoids torchcodec / FFmpeg
entirely, so the 'libtorchcodec' warning is irrelevant.

The output speaker tags are anonymous (SPEAKER_00, SPEAKER_01, ...). You do
NOT need to rename them: score_diarization.py maps speakers optimally
(Hungarian) when it computes DER, so the score is identical whatever the tags
are called. The names only matter if you want a human-readable release file.

Usage:
    python ambient_diarize.py --audio "G:\\...\\Run_3\\Run_3_Room.wav"
        --hf-token hf_xxx
        --out "G:\\...\\Run_3\\Run_3_Ambient_pyannote.txt"

    # optional: hint the speaker count (default: let pyannote decide)
    python ambient_diarize.py ... --num-speakers 5
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_in_memory(path: str, target_sr: int = 16000):
    """Read a WAV with soundfile, downmix to mono, resample — no torchcodec."""
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32, copy=False)
    wav = torch.from_numpy(data).unsqueeze(0)        # (1, num_samples)
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return {"waveform": wav, "sample_rate": sr}


def load_pipeline(model: str, token):
    from pyannote.audio import Pipeline
    # token kwarg name changed across versions; try the modern one first.
    # token may be None -> pyannote uses the credential stored by
    # `huggingface-cli login`, which is the recommended path.
    last_err = None
    attempts = [{"use_auth_token": token}, {"token": token}]
    if token is None:
        attempts.append({})  # rely entirely on stored login
    for kw in attempts:
        try:
            pipe = Pipeline.from_pretrained(model, **kw)
            if pipe is not None:
                return pipe
        except TypeError as e:
            last_err = e
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not load pipeline '{model}'. Run check_hf_access.py first — it "
        f"will tell you whether this is a login, gate, or network issue. "
        f"Underlying error:\n{last_err}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--audio", required=True, help="ambient WAV (e.g. Run_3_Room.wav)")
    p.add_argument("--hf-token", default=None,
                   help="HF token (optional; by default uses your "
                        "`huggingface-cli login` credential)")
    p.add_argument("--out", help="output Audacity label file "
                                 "(default: <audio>_pyannote.txt)")
    p.add_argument("--model", default="pyannote/speaker-diarization-3.1")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="hint exact speaker count; omit to let pyannote decide")
    p.add_argument("--reuse-cache", action="store_true",
                   help="if a cached result exists for this output, load it and "
                        "skip the model + the ~25 min compute")
    a = p.parse_args(argv)

    audio_path = Path(a.audio)
    if not audio_path.is_file():
        print(f"ERROR: audio not found: {audio_path}", file=sys.stderr)
        return 2
    out_path = Path(a.out) if a.out else audio_path.with_name(
        audio_path.stem + "_pyannote.txt")
    cache_path = out_path.with_suffix(".cache.pkl")

    import pickle

    # --- Fast path: reuse a previous compute if asked and available ----------
    segments = None
    if a.reuse_cache and cache_path.is_file():
        try:
            with cache_path.open("rb") as fh:
                segments = pickle.load(fh)
            print(f"Reusing cached diarization ({len(segments)} segments) from "
                  f"{cache_path.name} — skipping the model and the ~25 min compute.")
        except Exception as e:
            print(f"Could not read cache ({e}); recomputing.")
            segments = None

    if segments is None:
        print(f"Loading pipeline '{a.model}' (first run downloads the models)…")
        pipeline = load_pipeline(a.model, a.hf_token)
        pipeline.to(torch.device("cpu"))

        print(f"Loading audio in memory: {audio_path.name}")
        audio = load_in_memory(str(audio_path))
        dur = audio["waveform"].shape[1] / audio["sample_rate"]
        print(f"  duration {dur:.1f} s — running on CPU, this can take "
              f"10–30 min for a ~30 min file. Please wait…")

        kw = {}
        if a.num_speakers is not None:
            kw["num_speakers"] = a.num_speakers

        t0 = time.time()
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                diarization = pipeline(audio, hook=hook, **kw)
        except Exception:
            # progress hook is cosmetic; fall back to a plain call
            diarization = pipeline(audio, **kw)
        print(f"Diarization finished in {time.time() - t0:.0f} s.")

        # --- Normalize the result to a pyannote Annotation -------------------
        # Different pyannote versions return either an Annotation directly or a
        # wrapper (e.g. DiarizeOutput) holding it. Find it whatever it's called.
        annotation = diarization
        if not hasattr(annotation, "itertracks"):
            found = None
            for attr in ("speaker_diarization", "diarization", "annotation", "output"):
                cand = getattr(annotation, attr, None)
                if cand is not None and hasattr(cand, "itertracks"):
                    found = cand
                    break
            if found is None:
                for name in dir(annotation):
                    if name.startswith("_"):
                        continue
                    cand = getattr(annotation, name, None)
                    if hasattr(cand, "itertracks"):
                        found = cand
                        break
            if found is None:
                raise RuntimeError(
                    "Could not locate an Annotation with .itertracks() in the "
                    f"pipeline output of type {type(diarization).__name__}. "
                    f"Available attributes: "
                    f"{[x for x in dir(diarization) if not x.startswith('_')]}"
                )
            annotation = found

        # Extract a plain, always-picklable list of segments and CACHE IT NOW,
        # before any file writing can fail — so the expensive compute is safe.
        segments = [(float(t.start), float(t.end), str(spk))
                    for t, _, spk in annotation.itertracks(yield_label=True)]
        try:
            with cache_path.open("wb") as fh:
                pickle.dump(segments, fh)
            print(f"Cached result -> {cache_path.name} "
                  f"(re-run with --reuse-cache to skip recompute).")
        except Exception as e:
            print(f"(could not write cache: {e} — not fatal)")

    # --- Write Audacity labels + per-speaker summary (from segments) ---------
    spk_time = {}
    with out_path.open("w", encoding="utf-8") as f:
        for start, end, speaker in segments:
            f.write(f"{start:.3f}\t{end:.3f}\t{speaker}\n")
            spk_time[speaker] = spk_time.get(speaker, 0.0) + (end - start)

    print(f"\nWrote {len(segments)} segments -> {out_path}")
    print(f"pyannote found {len(spk_time)} speakers:")
    for spk in sorted(spk_time, key=lambda s: -spk_time[s]):
        print(f"   {spk:<14} {spk_time[spk]:7.1f} s")
    print("\nNow score it (DER is mapping-invariant, so anonymous tags are fine).")
    return 0


if __name__ == "__main__":
    sys.exit(main())