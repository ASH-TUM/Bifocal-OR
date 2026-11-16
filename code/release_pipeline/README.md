# Release pipeline (provenance)

These scripts built the public release from the raw capture. They are here to document how the released offsets, frame times and metadata were produced; you don't need them to use the dataset. They read the internal raw layout, which is not public in this form.

| Script | What it did |
| --- | --- |
| `collect_sync_files.py` | collected the small timing, synchronization and calibration files (frame timestamps, microscope encoder logs, audio-sync projects) from the raw capture |
| `sync_audio_feeds.py` | placed each microscope recording on the audio timeline by cross-correlating its microphone with the synchronized ambient track, and measured residual offsets of the wearable tracks |
| `build_timing.py` | wrote `sync/`: per-run offsets, room-camera frame times and microscope frame times (from the encoder progress log); clap-verified offsets via `--verified` |
| `build_release_metadata.py` | built the metadata layer (annotations, calibration, run table, events, README) and checked it against the paper's tables |
| `upload_release.py` + `release_manifest.csv` | packed the frame folders into one `.tar` per run, camera and modality and uploaded everything to Hugging Face (resumable; refuses to upload into an ungated public repository) |

The time bases are documented in the dataset card on Hugging Face: room-camera timeline, audio timeline, and microscope frame times.
