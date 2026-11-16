# Bifocal-OR: A Dual-Scale Multi-Sensor Dataset for Ophthalmic Surgery

Bifocal-OR records realistically simulated ophthalmic microsurgery (epiretinal membrane peeling on eye phantoms and ex-vivo porcine tissue) on one shared timeline:

- five room-wide RGB-D cameras;
- the microscope-mounted camera;
- an ambient room microphone;
- wearable smartphone microphones.

It comprises nine runs (5.8 h of room recording), with phase and speaker annotations, per-frame timestamps, camera calibrations and an inventory of the unscripted events it preserves.

- **Dataset:** <https://huggingface.co/datasets/AngeloHenriques/Bifocal-OR>. Access is gated (research use; request access on the page). License: CC BY-NC 4.0.
- **Paper:** A. Henriques et al., *Bifocal-OR: A Dual-Scale Multi-Sensor Dataset for Ophthalmic Surgery*, under review. The arXiv link follows.

This repository holds the code used for the paper.

| Folder | Content |
| --- | --- |
| [`code/phase_recognition`](code/phase_recognition) | TeCNO baselines (Stage 1 ResNet-50, Stage 2 MS-TCN), leave-one-out protocol, view selection and fusion, evaluation, per-frame test predictions |
| [`code/diarization`](code/diarization) | wearable pseudo-labeling pipeline, far-field pyannote baseline, DER scoring |
| [`code/figures`](code/figures) | scripts for Figs. 3–6 of the paper |
| [`code/release_pipeline`](code/release_pipeline) | how the release was built from the raw capture (synchronization, metadata, upload); provenance only |

Each folder has a README with the commands used for the paper.

## License

The code is released under GPL-3.0 (see `LICENSE`). The dataset has its own license (CC BY-NC 4.0) and access terms on Hugging Face.

## Citation

Will be added after publication.
