# Speaker diarization

These are the scripts behind the speaker labels and the diarization benchmark (paper Sec. III-B, IV-B, Table V).

Requirements: Python ≥ 3.10, `numpy`, `scipy`, `soundfile`, `torch`, `torchaudio`.
- The pseudo-labeling pipeline loads Silero VAD via `torch.hub`, which needs internet on the first run.
- The far-field baseline needs `pyannote.audio==4.0.4` and a Hugging Face account that has accepted the terms of `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

All label files are Audacity label files: `start<TAB>end<TAB>speaker`.

## Folder layout the scripts expect

```
<DVR_Sync>/Run_<n>/Run_<n>_<Role>.wav      wearable tracks (Surgeon, OTA, Runner, Anesthesiologist, Nurse, Spectator)
<DVR_Sync>/Run_<n>/Run_<n>_Room.wav        ambient track
```

In the Hugging Face release, the same files are `runs/runNN/audio/<role>.wav` and `runs/runNN/audio/room.wav`. Copy or rename them into the layout above before running the scripts.

## 1 · Wearable pseudo-labels (`auto_diarize_v2.py`)

Steps: bandpass filtering, Silero VAD, raw-energy contrast referee between the wearable microphones, and floor-hold median filtering.

Settings used for the paper:

```bash
python auto_diarize_v2.py --base-dir <DVR_Sync> --vad-threshold 0.35 --gain-clip 1.0 --min-contrast-db 1.0
```

- `--gain-clip 1.0` means raw energy, without per-microphone gain equalization.
- The script writes `Run_<n>_AntiFragLabels.txt`, a per-label confidence CSV and `Run_<n>_diarize_summary.json`, which also records the settings.

Thresholds were tuned on Run 3 with `tune_thresholds.py`:

```bash
python tune_thresholds.py --script auto_diarize_v2.py --scorer score_diarization.py \
    --base-dir <DVR_Sync> --run Run_3 \
    --gt <DVR_Sync>/Run_3/Run_3_Labels_GroundTruth.txt \
    --hyp <DVR_Sync>/Run_3/Run_3_AntiFragLabels.txt --ignore-speaker Anesthesiologist
```

## 2 · Far-field baseline on the ambient microphone (`ambient_diarize.py`)

```bash
python ambient_diarize.py --audio <DVR_Sync>/Run_3/Run_3_Room.wav --out Run_3_Ambient_pyannote.txt
python ambient_diarize.py --audio <DVR_Sync>/Run_3/Run_3_Room.wav --num-speakers 5 --out Run_3_pyannote_n5.txt   # oracle speaker count
```

This runs the untuned `pyannote/speaker-diarization-3.1` pipeline on the CPU, which takes about 10–30 min per run.

## 3 · Scoring

- `score_diarization.py REF HYP --collar 0.25` prints DER with its components, a confusion matrix and per-speaker routing. Install `pyannote.metrics` to get the collar-aware number; without it the script reports the frame-level DER without a collar.
- `score_oracle.py --sync <DVR_Sync> --new <folder>` scores with the exact convention of Table V: 250 ms collar, optimal mapping, overlap-aware, Run 3 without the Anesthesiologist (dead wearable channel). It reproduces Table V from the original pyannote outputs and also scores the oracle-count runs.

## 4 · Residual synchronization (`measure_drift.py`)

```bash
python measure_drift.py --run-dir <DVR_Sync>/Run_4
```

The script cross-correlates each wearable track with the ambient track in windows spread across the run and reports the residual offsets.
