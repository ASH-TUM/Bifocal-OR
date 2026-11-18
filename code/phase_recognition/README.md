# Multi-View Surgical Phase Recognition for Vitrectomy

Surgical phase recognition in an ophthalmic operating room, recorded by **five
ceiling/room cameras**. The pipeline builds on
[TeCNO](https://github.com/tobiascz/TeCNO) (Czempiel et al., MICCAI 2020), a
ResNet-50 frame encoder followed by a causal multi-stage temporal convolutional
network. It extends it with a **label-free multi-camera ensemble** that picks
the most reliable views at inference time.

## What this repo adds on top of TeCNO

- **Vitrectomy dataset adaptation.** Five OR workflow phases (Room_Prep, Patient_Entry,
  Sterile_Draping, Microsurgery, Emergence), 9 recorded runs × 5 cameras, 1 fps.
- **Leakage-free evaluation protocol.** Deterministic leave-one-out splits
  shared by both stages (`datasets/loo_splits.py`).
  - *Experiment A:* 8-fold in-domain LOO on Runs 1–8 (plastic phantom eye). Per fold:
    6 runs train, 1 run validation (the next run, cyclically), 1 run test.
  - *Experiment B:* zero-shot out-of-distribution test on Run 9 (porcine eye), which is
    never used for training or validation. Each of the 8 fold models is applied to it.
- **Multi-camera ensembling** (`evaluation/loo_voting.py`):
  - static hard / soft voting over all camera subsets
  - **dynamic top-k voting**: cameras are ranked by softmax entropy (no labels
    needed), and the k most confident ones vote, weighted by 1/entropy. Ranking can be
    per frame, over a sliding window, or once per video.
  - sensor-dropout robustness analysis and per-frame vs. per-video diagnostics
    (boundary vs. interior accuracy)
- **Segmental metrics** (Lea et al., 2017): Edit score and F1@{10,25,50} alongside
  frame accuracy (`evaluation/loo_aggregate.py`).
- **Compute-free analysis.** Inference writes one small `{y_true, y_pred, y_soft}`
  pickle per camera and fold. Every ensemble experiment runs on these files in seconds,
  without a GPU.
- Port to PyTorch Lightning 1.9 / PyTorch 2.x.

## Results

All numbers come from the shipped `predictions/` and can be reproduced in seconds with the
commands in [Analysis](#analysis). They use the **timestamp-exact labels** (November 2026
re-run, see [Labels and frame timing](#labels-and-frame-timing)). Frame accuracy and macro F1
are pooled over all frames of all folds; Edit and F1@k are means over the 8 folds.

Protocol: Stage 1 and Stage 2 were trained end to end once per camera and fold
(`run_expAB.sh`, Stage-1 seed 1234 + fold, Stage-2 seed 5678 + fold). Stage 2 was then retrained
twice more on the same Stage-1 features with seeds 6678 and 7678 (`run_stage2_refit.sh`).
**± SD** is the sample standard deviation over these three Stage-2 seeds. Ensemble = dynamic
top-3 voting, cameras ranked by softmax entropy (no labels used).

### Headline: mean ± SD over three Stage-2 seeds

| Setting | Method | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| In-domain (Exp. A) | best single camera (camera05) | 0.916 ± 0.010 | 0.912 ± 0.012 | 24.9 ± 3.2 | 37.0 ± 2.8 | 36.9 ± 3.0 | 35.4 ± 2.6 |
| In-domain (Exp. A) | dynamic top-3, per video | 0.902 ± 0.033 | 0.897 ± 0.036 | 38.9 ± 5.3 | 51.8 ± 5.7 | 51.4 ± 5.5 | 50.0 ± 5.5 |
| OOD zero-shot (Exp. B) | best single camera (camera03) | 0.920 ± 0.005 | 0.914 ± 0.002 | 26.5 ± 1.7 | 39.5 ± 1.6 | 39.5 ± 1.6 | 38.6 ± 1.3 |
| OOD zero-shot (Exp. B) | dynamic top-3, per video | 0.919 ± 0.023 | 0.908 ± 0.020 | 32.0 ± 1.6 | 44.9 ± 2.8 | 44.3 ± 2.2 | 43.6 ± 2.1 |

Main observations:

- **Ensemble vs. single cameras.** Ranked per video, the ensemble reaches 0.902 in-domain and
  0.919 zero-shot. Zero-shot it equals the best single camera (camera03, 0.920); in-domain it stays
  1.4 points below the best single camera (camera05, 0.916). It does this without knowing which
  camera is best, and the best camera is not the same in both settings.
- **Segments.** Ranking over a window or per video gives cleaner segments than the best single
  camera in both settings (per video: Edit 38.9 vs. 24.9 in-domain, 32.0 vs. 26.5 zero-shot). The
  smoothed per-frame ranking (Δ = ±5 s) gives the cleanest phase blocks: Edit 69.7 / F1@50 77.4
  in-domain and 43.2 / 53.0 zero-shot, at 2.8 / 1.7 points lower frame accuracy than per-video
  ranking.
- **Seed sensitivity.** Single-camera accuracy varies by at most ±1.1 points across Stage-2 seeds.
  The per-video ensemble varies more (±2–3 points) because the selected cameras change with the seed.
  Segmental metrics are the most seed-sensitive numbers (see the per-fold segment counts below).
- **Weak camera.** camera02 is the weakest camera in-domain (0.717). The per-video ensemble gives it
  0–13 % of the votes in-domain, against 20 % for a uniform choice.
- **Zero-shot failure mode.** camera01 and camera02 drop to 0.48–0.69 on Run 9 in folds 5 and 6 for
  all three seeds: 63–97 % of the Microsurgery frames of Run 9 are predicted as Emergence. Because it does not depend on the
  Stage-2 seed, it comes from the Stage-1 features of these two folds.
- **Against the previous release** (constant-rate labels, last table below): in-domain accuracy is
  essentially unchanged (ensemble +0.003, camera05 +0.028, camera01 −0.026). Zero-shot accuracy
  is lower for every camera (−0.015 to −0.146) and for the ensemble (−0.046). Segmental metrics
  are lower throughout, because predictions flicker more around phase boundaries.

### In-domain (Exp. A): every camera and every ensemble mode (mean ± SD over seeds)

| Method | Mean fold acc. | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| camera01 | 0.849 ± 0.002 | 0.844 ± 0.002 | 0.840 ± 0.003 | 43.5 ± 10.5 | 55.7 ± 9.7 | 53.9 ± 9.9 | 53.3 ± 10.4 |
| camera02 | 0.713 ± 0.007 | 0.717 ± 0.005 | 0.681 ± 0.011 | 22.4 ± 3.6 | 31.1 ± 5.5 | 26.8 ± 4.8 | 23.4 ± 4.7 |
| camera03 | 0.896 ± 0.008 | 0.896 ± 0.009 | 0.893 ± 0.010 | 27.3 ± 8.3 | 39.9 ± 9.0 | 39.9 ± 9.0 | 38.3 ± 9.2 |
| camera04 | 0.872 ± 0.002 | 0.874 ± 0.002 | 0.862 ± 0.003 | 30.4 ± 4.8 | 43.6 ± 4.3 | 43.1 ± 4.3 | 41.7 ± 4.6 |
| camera05 | 0.915 ± 0.008 | 0.916 ± 0.010 | 0.912 ± 0.012 | 24.9 ± 3.2 | 37.0 ± 2.8 | 36.9 ± 3.0 | 35.4 ± 2.6 |
| dynamic top-3, per frame (Δ=0) | 0.876 ± 0.008 | 0.874 ± 0.012 | 0.865 ± 0.010 | 41.7 ± 5.9 | 55.4 ± 5.8 | 54.6 ± 5.2 | 53.7 ± 6.8 |
| dynamic top-3, ±5 s window (Δ=5) | 0.876 ± 0.008 | 0.874 ± 0.012 | 0.865 ± 0.010 | 69.7 ± 3.8 | 79.2 ± 2.9 | 78.0 ± 2.2 | 77.4 ± 3.1 |
| dynamic top-3, per video (Δ=∞) | 0.904 ± 0.029 | 0.902 ± 0.033 | 0.897 ± 0.036 | 38.9 ± 5.3 | 51.8 ± 5.7 | 51.4 ± 5.5 | 50.0 ± 5.5 |

### OOD zero-shot (Exp. B): every camera and every ensemble mode (mean ± SD over seeds)

| Method | Mean fold acc. | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|
| camera01 | 0.791 ± 0.004 | 0.791 ± 0.004 | 0.793 ± 0.009 | 18.0 ± 1.0 | 29.1 ± 1.7 | 28.0 ± 1.8 | 23.0 ± 2.4 |
| camera02 | 0.793 ± 0.011 | 0.793 ± 0.011 | 0.814 ± 0.013 | 17.2 ± 4.4 | 27.4 ± 6.0 | 26.5 ± 5.6 | 22.9 ± 4.7 |
| camera03 | 0.920 ± 0.005 | 0.920 ± 0.005 | 0.914 ± 0.002 | 26.5 ± 1.7 | 39.5 ± 1.6 | 39.5 ± 1.6 | 38.6 ± 1.3 |
| camera04 | 0.856 ± 0.008 | 0.856 ± 0.008 | 0.823 ± 0.007 | 12.1 ± 2.1 | 21.1 ± 3.2 | 20.4 ± 3.8 | 19.0 ± 4.1 |
| camera05 | 0.871 ± 0.006 | 0.871 ± 0.006 | 0.849 ± 0.011 | 18.3 ± 4.2 | 29.1 ± 4.9 | 28.5 ± 4.9 | 24.6 ± 4.7 |
| dynamic top-3, per frame (Δ=0) | 0.902 ± 0.024 | 0.902 ± 0.024 | 0.891 ± 0.021 | 23.1 ± 4.6 | 34.9 ± 5.1 | 34.4 ± 4.9 | 33.0 ± 5.4 |
| dynamic top-3, ±5 s window (Δ=5) | 0.902 ± 0.025 | 0.902 ± 0.025 | 0.891 ± 0.022 | 43.2 ± 8.5 | 57.0 ± 7.2 | 55.8 ± 6.9 | 53.0 ± 6.8 |
| dynamic top-3, per video (Δ=∞) | 0.919 ± 0.023 | 0.919 ± 0.023 | 0.908 ± 0.020 | 32.0 ± 1.6 | 44.9 ± 2.8 | 44.3 ± 2.2 | 43.6 ± 2.1 |

<details>
<summary><b>Every seed separately</b></summary>

**In-domain (Exp. A)**

| Seed | Method | Mean fold acc. ± SD | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 0.847 ± 0.102 | 0.842 | 0.838 | 37.9 | 49.4 | 47.7 | 46.5 |
| 5678 | camera02 | 0.712 ± 0.161 | 0.715 | 0.678 | 20.8 | 29.5 | 25.5 | 22.2 |
| 5678 | camera03 | 0.896 ± 0.046 | 0.897 | 0.896 | 17.8 | 29.4 | 29.4 | 27.6 |
| 5678 | camera04 | 0.874 ± 0.090 | 0.876 | 0.865 | 25.3 | 38.9 | 38.4 | 36.5 |
| 5678 | camera05 | 0.924 ± 0.035 | 0.926 | 0.925 | 25.9 | 39.4 | 39.4 | 37.9 |
| 5678 | top-3 per frame (Δ=0) | 0.876 ± 0.063 | 0.876 | 0.868 | 40.9 | 55.5 | 55.1 | 55.1 |
| 5678 | top-3 ±5 s window (Δ=5) | 0.876 ± 0.063 | 0.876 | 0.868 | 68.8 | 79.1 | 78.3 | 78.3 |
| 5678 | top-3 per video (Δ=∞) | 0.917 ± 0.056 | 0.918 | 0.916 | 32.7 | 45.3 | 45.3 | 43.9 |
| 6678 | camera01 | 0.850 ± 0.119 | 0.843 | 0.838 | 55.6 | 66.8 | 65.2 | 65.2 |
| 6678 | camera02 | 0.721 ± 0.154 | 0.723 | 0.693 | 26.5 | 37.3 | 32.1 | 28.6 |
| 6678 | camera03 | 0.887 ± 0.050 | 0.888 | 0.883 | 33.1 | 44.7 | 44.7 | 43.7 |
| 6678 | camera04 | 0.873 ± 0.085 | 0.872 | 0.863 | 31.2 | 44.6 | 44.1 | 43.1 |
| 6678 | camera05 | 0.907 ± 0.045 | 0.906 | 0.901 | 27.6 | 37.7 | 37.7 | 35.7 |
| 6678 | top-3 per frame (Δ=0) | 0.868 ± 0.099 | 0.862 | 0.854 | 48.0 | 61.2 | 59.7 | 59.7 |
| 6678 | top-3 ±5 s window (Δ=5) | 0.868 ± 0.099 | 0.862 | 0.854 | 73.9 | 82.1 | 80.1 | 80.1 |
| 6678 | top-3 per video (Δ=∞) | 0.871 ± 0.114 | 0.863 | 0.855 | 42.3 | 54.6 | 53.3 | 52.1 |
| 7678 | camera01 | 0.850 ± 0.109 | 0.846 | 0.843 | 37.0 | 50.8 | 48.6 | 48.1 |
| 7678 | camera02 | 0.706 ± 0.147 | 0.712 | 0.671 | 19.9 | 26.7 | 22.7 | 19.5 |
| 7678 | camera03 | 0.904 ± 0.043 | 0.905 | 0.901 | 30.9 | 45.5 | 45.5 | 43.6 |
| 7678 | camera04 | 0.871 ± 0.120 | 0.875 | 0.859 | 34.7 | 47.3 | 46.9 | 45.5 |
| 7678 | camera05 | 0.914 ± 0.043 | 0.915 | 0.910 | 21.4 | 34.0 | 33.6 | 32.6 |
| 7678 | top-3 per frame (Δ=0) | 0.884 ± 0.077 | 0.885 | 0.874 | 36.3 | 49.6 | 49.2 | 46.3 |
| 7678 | top-3 ±5 s window (Δ=5) | 0.884 ± 0.075 | 0.885 | 0.874 | 66.4 | 76.3 | 75.6 | 73.9 |
| 7678 | top-3 per video (Δ=∞) | 0.923 ± 0.044 | 0.923 | 0.919 | 41.6 | 55.7 | 55.7 | 54.2 |

**OOD zero-shot (Exp. B)**

| Seed | Method | Mean fold acc. ± SD | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 0.794 ± 0.173 | 0.794 | 0.801 | 17.9 | 29.8 | 28.2 | 22.2 |
| 5678 | camera02 | 0.805 ± 0.162 | 0.805 | 0.827 | 20.9 | 32.7 | 30.7 | 26.4 |
| 5678 | camera03 | 0.920 ± 0.050 | 0.920 | 0.915 | 26.8 | 39.7 | 39.7 | 39.3 |
| 5678 | camera04 | 0.847 ± 0.077 | 0.847 | 0.815 | 10.4 | 18.2 | 17.1 | 15.9 |
| 5678 | camera05 | 0.876 ± 0.034 | 0.876 | 0.858 | 20.0 | 32.1 | 31.0 | 28.1 |
| 5678 | top-3 per frame (Δ=0) | 0.878 ± 0.058 | 0.878 | 0.869 | 23.0 | 33.7 | 33.1 | 31.0 |
| 5678 | top-3 ±5 s window (Δ=5) | 0.877 ± 0.058 | 0.877 | 0.868 | 43.4 | 56.3 | 54.9 | 50.6 |
| 5678 | top-3 per video (Δ=∞) | 0.894 ± 0.101 | 0.894 | 0.888 | 33.7 | 48.1 | 46.7 | 45.9 |
| 6678 | camera01 | 0.791 ± 0.167 | 0.791 | 0.795 | 19.1 | 30.3 | 29.7 | 25.8 |
| 6678 | camera02 | 0.791 ± 0.166 | 0.791 | 0.813 | 18.6 | 28.7 | 28.5 | 24.9 |
| 6678 | camera03 | 0.915 ± 0.035 | 0.915 | 0.911 | 24.6 | 37.8 | 37.8 | 37.2 |
| 6678 | camera04 | 0.861 ± 0.080 | 0.861 | 0.830 | 14.5 | 24.5 | 24.5 | 23.6 |
| 6678 | camera05 | 0.873 ± 0.028 | 0.873 | 0.851 | 13.5 | 23.5 | 22.8 | 19.3 |
| 6678 | top-3 per frame (Δ=0) | 0.904 ± 0.066 | 0.904 | 0.891 | 18.6 | 30.4 | 30.4 | 28.9 |
| 6678 | top-3 ±5 s window (Δ=5) | 0.905 ± 0.066 | 0.905 | 0.893 | 34.6 | 50.2 | 49.3 | 47.7 |
| 6678 | top-3 per video (Δ=∞) | 0.923 ± 0.047 | 0.923 | 0.908 | 32.0 | 43.4 | 42.9 | 42.4 |
| 7678 | camera01 | 0.786 ± 0.145 | 0.786 | 0.783 | 17.2 | 27.1 | 26.1 | 21.2 |
| 7678 | camera02 | 0.783 ± 0.165 | 0.783 | 0.801 | 12.3 | 20.9 | 20.2 | 17.6 |
| 7678 | camera03 | 0.925 ± 0.038 | 0.925 | 0.916 | 28.1 | 41.0 | 41.0 | 39.4 |
| 7678 | camera04 | 0.861 ± 0.050 | 0.861 | 0.824 | 11.5 | 20.5 | 19.5 | 17.5 |
| 7678 | camera05 | 0.864 ± 0.072 | 0.864 | 0.837 | 21.4 | 31.7 | 31.7 | 26.3 |
| 7678 | top-3 per frame (Δ=0) | 0.925 ± 0.047 | 0.925 | 0.911 | 27.8 | 40.4 | 39.8 | 39.1 |
| 7678 | top-3 ±5 s window (Δ=5) | 0.926 ± 0.047 | 0.926 | 0.913 | 51.5 | 64.5 | 63.1 | 60.7 |
| 7678 | top-3 per video (Δ=∞) | 0.940 ± 0.039 | 0.940 | 0.928 | 30.5 | 43.1 | 43.1 | 42.3 |

</details>

<details>
<summary><b>Camera selection rate of the dynamic ensemble</b></summary>

Share of all votes per camera, in %. A uniform choice would give every camera 20 %.

| Setting | Mode | Seed | camera01 | camera02 | camera03 | camera04 | camera05 |
|---|---|---|---|---|---|---|---|
| In-domain (Exp. A) | per frame (Δ=0) | 5678 | 19.6 | 15.7 | 18.2 | 22.8 | 23.6 |
| In-domain (Exp. A) | per frame (Δ=0) | 6678 | 18.3 | 16.2 | 20.2 | 22.0 | 23.2 |
| In-domain (Exp. A) | per frame (Δ=0) | 7678 | 21.1 | 12.9 | 19.1 | 24.3 | 22.6 |
| In-domain (Exp. A) | ±5 s window (Δ=5) | 5678 | 19.6 | 15.6 | 18.2 | 23.0 | 23.6 |
| In-domain (Exp. A) | ±5 s window (Δ=5) | 6678 | 18.3 | 16.2 | 20.2 | 22.1 | 23.2 |
| In-domain (Exp. A) | ±5 s window (Δ=5) | 7678 | 21.0 | 12.9 | 19.1 | 24.5 | 22.5 |
| In-domain (Exp. A) | per video (Δ=∞) | 5678 | 24.5 | 0.0 | 29.8 | 28.7 | 16.9 |
| In-domain (Exp. A) | per video (Δ=∞) | 6678 | 29.4 | 13.4 | 15.1 | 29.8 | 12.3 |
| In-domain (Exp. A) | per video (Δ=∞) | 7678 | 24.5 | 8.3 | 28.6 | 25.2 | 13.4 |
| OOD zero-shot (Exp. B) | per frame (Δ=0) | 5678 | 14.8 | 18.2 | 23.9 | 19.7 | 23.5 |
| OOD zero-shot (Exp. B) | per frame (Δ=0) | 6678 | 16.6 | 18.3 | 22.4 | 21.5 | 21.2 |
| OOD zero-shot (Exp. B) | per frame (Δ=0) | 7678 | 15.0 | 16.7 | 25.0 | 20.7 | 22.6 |
| OOD zero-shot (Exp. B) | ±5 s window (Δ=5) | 5678 | 14.8 | 18.1 | 23.9 | 19.7 | 23.4 |
| OOD zero-shot (Exp. B) | ±5 s window (Δ=5) | 6678 | 16.6 | 18.3 | 22.3 | 21.6 | 21.1 |
| OOD zero-shot (Exp. B) | ±5 s window (Δ=5) | 7678 | 15.0 | 16.7 | 25.0 | 20.7 | 22.5 |
| OOD zero-shot (Exp. B) | per video (Δ=∞) | 5678 | 8.3 | 20.8 | 29.2 | 16.7 | 25.0 |
| OOD zero-shot (Exp. B) | per video (Δ=∞) | 6678 | 12.5 | 20.8 | 29.2 | 20.8 | 16.7 |
| OOD zero-shot (Exp. B) | per video (Δ=∞) | 7678 | 16.7 | 25.0 | 29.2 | 4.2 | 25.0 |

</details>

<details>
<summary><b>Per-class F1, precision and recall</b></summary>

**Per-class F1, In-domain (Exp. A), mean ± SD over seeds**

| Method | Room_Prep | Patient_Entry | Sterile_Draping | Microsurgery | Emergence |
|---|---|---|---|---|---|
| camera01 | 0.954 ± 0.006 | 0.860 ± 0.011 | 0.761 ± 0.003 | 0.931 ± 0.006 | 0.692 ± 0.004 |
| camera02 | 0.747 ± 0.016 | 0.580 ± 0.046 | 0.634 ± 0.004 | 0.952 ± 0.001 | 0.489 ± 0.010 |
| camera03 | 0.952 ± 0.012 | 0.846 ± 0.004 | 0.839 ± 0.010 | 0.952 ± 0.004 | 0.877 ± 0.035 |
| camera04 | 0.943 ± 0.033 | 0.849 ± 0.009 | 0.800 ± 0.016 | 0.962 ± 0.002 | 0.758 ± 0.014 |
| camera05 | 0.964 ± 0.010 | 0.912 ± 0.008 | 0.861 ± 0.021 | 0.959 ± 0.002 | 0.864 ± 0.038 |
| top-3 per frame (Δ=0) | 0.958 ± 0.016 | 0.853 ± 0.006 | 0.790 ± 0.039 | 0.963 ± 0.003 | 0.763 ± 0.022 |
| top-3 ±5 s window (Δ=5) | 0.958 ± 0.016 | 0.852 ± 0.006 | 0.789 ± 0.039 | 0.963 ± 0.003 | 0.765 ± 0.023 |
| top-3 per video (Δ=∞) | 0.963 ± 0.006 | 0.881 ± 0.014 | 0.838 ± 0.067 | 0.963 ± 0.002 | 0.839 ± 0.098 |

Per-class precision / recall, In-domain (Exp. A), mean over seeds:

| Method | Room_Prep P / R | Patient_Entry P / R | Sterile_Draping P / R | Microsurgery P / R | Emergence P / R |
|---|---|---|---|---|---|
| camera01 | 0.920 / 0.991 | 0.973 / 0.771 | 0.801 / 0.726 | 0.973 / 0.891 | 0.555 / 0.921 |
| camera02 | 0.868 / 0.655 | 0.901 / 0.428 | 0.684 / 0.592 | 0.977 / 0.929 | 0.348 / 0.827 |
| camera03 | 0.913 / 0.994 | 0.895 / 0.803 | 0.837 / 0.841 | 0.968 / 0.935 | 0.819 / 0.946 |
| camera04 | 0.905 / 0.985 | 0.940 / 0.774 | 0.850 / 0.756 | 0.965 / 0.960 | 0.648 / 0.914 |
| camera05 | 0.931 / 1.000 | 0.907 / 0.917 | 0.906 / 0.822 | 0.966 / 0.951 | 0.812 / 0.926 |
| top-3 per video | 0.934 / 0.994 | 0.957 / 0.817 | 0.845 / 0.836 | 0.976 / 0.951 | 0.776 / 0.932 |

**Per-class F1, OOD zero-shot (Exp. B), mean ± SD over seeds**

| Method | Room_Prep | Patient_Entry | Sterile_Draping | Microsurgery | Emergence |
|---|---|---|---|---|---|
| camera01 | 0.942 ± 0.032 | 0.771 ± 0.006 | 0.825 ± 0.022 | 0.870 ± 0.008 | 0.558 ± 0.006 |
| camera02 | 0.897 ± 0.016 | 0.854 ± 0.016 | 0.875 ± 0.015 | 0.796 ± 0.008 | 0.647 ± 0.011 |
| camera03 | 0.952 ± 0.010 | 0.878 ± 0.004 | 0.878 ± 0.008 | 0.948 ± 0.009 | 0.914 ± 0.006 |
| camera04 | 0.942 ± 0.016 | 0.703 ± 0.010 | 0.786 ± 0.018 | 0.968 ± 0.004 | 0.716 ± 0.028 |
| camera05 | 0.943 ± 0.022 | 0.839 ± 0.018 | 0.752 ± 0.013 | 0.984 ± 0.001 | 0.726 ± 0.004 |
| top-3 per frame (Δ=0) | 0.953 ± 0.005 | 0.864 ± 0.018 | 0.862 ± 0.030 | 0.960 ± 0.016 | 0.814 ± 0.046 |
| top-3 ±5 s window (Δ=5) | 0.954 ± 0.005 | 0.864 ± 0.019 | 0.863 ± 0.032 | 0.960 ± 0.016 | 0.816 ± 0.048 |
| top-3 per video (Δ=∞) | 0.948 ± 0.010 | 0.868 ± 0.019 | 0.905 ± 0.029 | 0.958 ± 0.022 | 0.861 ± 0.058 |

Per-class precision / recall, OOD zero-shot (Exp. B), mean over seeds:

| Method | Room_Prep P / R | Patient_Entry P / R | Sterile_Draping P / R | Microsurgery P / R | Emergence P / R |
|---|---|---|---|---|---|
| camera01 | 0.891 / 1.000 | 0.688 / 0.877 | 0.901 / 0.760 | 0.993 / 0.774 | 0.458 / 0.713 |
| camera02 | 0.841 / 0.961 | 0.915 / 0.801 | 0.882 / 0.868 | 0.990 / 0.665 | 0.490 / 0.951 |
| camera03 | 0.909 / 1.000 | 0.839 / 0.921 | 0.940 / 0.824 | 0.934 / 0.962 | 0.926 / 0.901 |
| camera04 | 0.891 / 1.000 | 0.690 / 0.718 | 0.879 / 0.711 | 0.951 / 0.985 | 0.690 / 0.745 |
| camera05 | 0.893 / 1.000 | 0.876 / 0.805 | 0.901 / 0.645 | 0.991 / 0.977 | 0.615 / 0.887 |
| top-3 per video | 0.902 / 1.000 | 0.863 / 0.876 | 0.932 / 0.881 | 0.985 / 0.933 | 0.805 / 0.926 |

</details>

<details>
<summary><b>Per fold: accuracy, Edit, F1@50 and number of predicted segments</b></summary>

Fold k tests Run k+1 (in-domain) or applies the fold-k model to Run 9 (OOD).

**In-domain (Exp. A), frame accuracy**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 0.786 | 0.779 | 0.930 | 0.920 | 0.959 | 0.710 | 0.949 | 0.744 |
| 5678 | camera02 | 0.710 | 0.882 | 0.725 | 0.445 | 0.915 | 0.566 | 0.824 | 0.631 |
| 5678 | camera03 | 0.815 | 0.871 | 0.955 | 0.906 | 0.915 | 0.924 | 0.851 | 0.930 |
| 5678 | camera04 | 0.782 | 0.959 | 0.941 | 0.734 | 0.966 | 0.795 | 0.913 | 0.899 |
| 5678 | camera05 | 0.888 | 0.967 | 0.902 | 0.934 | 0.954 | 0.920 | 0.872 | 0.956 |
| 5678 | top-3 per frame (Δ=0) | 0.849 | 0.918 | 0.924 | 0.832 | 0.955 | 0.755 | 0.896 | 0.881 |
| 5678 | top-3 ±5 s window (Δ=5) | 0.850 | 0.916 | 0.925 | 0.830 | 0.954 | 0.754 | 0.894 | 0.883 |
| 5678 | top-3 per video (Δ=∞) | 0.787 | 0.959 | 0.944 | 0.911 | 0.961 | 0.929 | 0.909 | 0.938 |
| 6678 | camera01 | 0.844 | 0.770 | 0.951 | 0.940 | 0.958 | 0.651 | 0.951 | 0.734 |
| 6678 | camera02 | 0.774 | 0.869 | 0.729 | 0.526 | 0.946 | 0.525 | 0.780 | 0.619 |
| 6678 | camera03 | 0.849 | 0.798 | 0.950 | 0.912 | 0.947 | 0.888 | 0.884 | 0.872 |
| 6678 | camera04 | 0.867 | 0.931 | 0.966 | 0.778 | 0.970 | 0.746 | 0.908 | 0.820 |
| 6678 | camera05 | 0.909 | 0.946 | 0.918 | 0.950 | 0.952 | 0.900 | 0.837 | 0.847 |
| 6678 | top-3 per frame (Δ=0) | 0.856 | 0.910 | 0.950 | 0.936 | 0.951 | 0.690 | 0.908 | 0.746 |
| 6678 | top-3 ±5 s window (Δ=5) | 0.855 | 0.910 | 0.951 | 0.936 | 0.950 | 0.690 | 0.906 | 0.746 |
| 6678 | top-3 per video (Δ=∞) | 0.855 | 0.919 | 0.962 | 0.953 | 0.960 | 0.660 | 0.923 | 0.735 |
| 7678 | camera01 | 0.845 | 0.709 | 0.934 | 0.935 | 0.956 | 0.710 | 0.951 | 0.762 |
| 7678 | camera02 | 0.754 | 0.780 | 0.728 | 0.406 | 0.897 | 0.623 | 0.795 | 0.668 |
| 7678 | camera03 | 0.853 | 0.852 | 0.961 | 0.869 | 0.929 | 0.942 | 0.930 | 0.896 |
| 7678 | camera04 | 0.832 | 0.969 | 0.949 | 0.615 | 0.968 | 0.806 | 0.934 | 0.892 |
| 7678 | camera05 | 0.865 | 0.945 | 0.904 | 0.866 | 0.955 | 0.947 | 0.962 | 0.869 |
| 7678 | top-3 per frame (Δ=0) | 0.852 | 0.950 | 0.919 | 0.743 | 0.949 | 0.839 | 0.969 | 0.851 |
| 7678 | top-3 ±5 s window (Δ=5) | 0.851 | 0.948 | 0.914 | 0.746 | 0.949 | 0.843 | 0.970 | 0.852 |
| 7678 | top-3 per video (Δ=∞) | 0.852 | 0.959 | 0.956 | 0.875 | 0.952 | 0.939 | 0.962 | 0.888 |

**In-domain (Exp. A), Edit score**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 11.6 | 14.3 | 55.6 | 100.0 | 29.4 | 22.7 | 38.5 | 31.2 |
| 5678 | camera02 | 9.3 | 9.6 | 29.4 | 14.3 | 13.5 | 23.8 | 41.7 | 25.0 |
| 5678 | camera03 | 10.6 | 16.1 | 26.3 | 14.3 | 10.2 | 17.2 | 35.7 | 11.9 |
| 5678 | camera04 | 27.8 | 21.7 | 55.6 | 19.2 | 23.8 | 10.6 | 15.6 | 27.8 |
| 5678 | camera05 | 9.4 | 38.5 | 38.5 | 21.7 | 27.8 | 14.3 | 11.4 | 45.5 |
| 5678 | top-3 per frame (Δ=0) | 55.6 | 55.6 | 71.4 | 33.3 | 38.5 | 8.8 | 22.7 | 41.7 |
| 5678 | top-3 ±5 s window (Δ=5) | 71.4 | 100.0 | 71.4 | 71.4 | 71.4 | 18.5 | 83.3 | 62.5 |
| 5678 | top-3 per video (Δ=∞) | 20.8 | 71.4 | 55.6 | 17.2 | 55.6 | 12.2 | 16.7 | 12.2 |
| 6678 | camera01 | 55.6 | 10.2 | 55.6 | 83.3 | 71.4 | 23.5 | 100.0 | 45.5 |
| 6678 | camera02 | 35.7 | 20.0 | 62.5 | 8.8 | 23.8 | 25.0 | 14.3 | 22.2 |
| 6678 | camera03 | 100.0 | 9.4 | 38.5 | 35.7 | 26.3 | 11.4 | 31.2 | 12.5 |
| 6678 | camera04 | 29.4 | 38.5 | 55.6 | 27.8 | 55.6 | 10.9 | 22.7 | 8.9 |
| 6678 | camera05 | 9.4 | 29.4 | 23.8 | 22.7 | 100.0 | 9.4 | 12.8 | 13.2 |
| 6678 | top-3 per frame (Δ=0) | 71.4 | 55.6 | 55.6 | 29.4 | 55.6 | 55.6 | 15.6 | 45.5 |
| 6678 | top-3 ±5 s window (Δ=5) | 100.0 | 100.0 | 71.4 | 71.4 | 71.4 | 55.6 | 50.0 | 71.4 |
| 6678 | top-3 per video (Δ=∞) | 100.0 | 23.8 | 45.5 | 55.6 | 55.6 | 12.1 | 12.2 | 33.3 |
| 7678 | camera01 | 71.4 | 10.9 | 25.0 | 38.5 | 45.5 | 16.7 | 38.5 | 50.0 |
| 7678 | camera02 | 18.5 | 8.2 | 41.7 | 20.8 | 15.2 | 8.1 | 7.9 | 38.5 |
| 7678 | camera03 | 71.4 | 29.4 | 23.8 | 23.8 | 29.4 | 23.8 | 25.0 | 20.8 |
| 7678 | camera04 | 19.2 | 55.6 | 55.6 | 8.8 | 71.4 | 15.6 | 35.7 | 15.6 |
| 7678 | camera05 | 8.8 | 23.8 | 29.4 | 8.5 | 21.7 | 23.8 | 41.7 | 13.2 |
| 7678 | top-3 per frame (Δ=0) | 71.4 | 38.5 | 21.7 | 19.2 | 23.8 | 8.5 | 71.4 | 35.7 |
| 7678 | top-3 ±5 s window (Δ=5) | 100.0 | 71.4 | 55.6 | 50.0 | 55.6 | 15.2 | 100.0 | 83.3 |
| 7678 | top-3 per video (Δ=∞) | 71.4 | 71.4 | 45.5 | 16.7 | 26.3 | 18.5 | 55.6 | 27.8 |

**In-domain (Exp. A), F1@50**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 16.7 | 15.0 | 71.4 | 100.0 | 45.5 | 29.6 | 55.6 | 38.1 |
| 5678 | camera02 | 10.2 | 14.0 | 36.4 | 12.1 | 23.8 | 15.4 | 47.1 | 19.0 |
| 5678 | camera03 | 15.4 | 22.2 | 41.7 | 20.0 | 18.5 | 29.4 | 52.6 | 21.3 |
| 5678 | camera04 | 34.8 | 35.7 | 71.4 | 25.8 | 38.5 | 15.4 | 27.0 | 43.5 |
| 5678 | camera05 | 13.8 | 55.6 | 55.6 | 35.7 | 43.5 | 20.0 | 16.3 | 62.5 |
| 5678 | top-3 per frame (Δ=0) | 71.4 | 71.4 | 83.3 | 50.0 | 55.6 | 12.9 | 37.0 | 58.8 |
| 5678 | top-3 ±5 s window (Δ=5) | 83.3 | 100.0 | 83.3 | 83.3 | 83.3 | 25.0 | 90.9 | 76.9 |
| 5678 | top-3 per video (Δ=∞) | 27.6 | 83.3 | 71.4 | 29.4 | 71.4 | 17.4 | 28.6 | 21.7 |
| 6678 | camera01 | 71.4 | 18.5 | 71.4 | 90.9 | 83.3 | 36.4 | 100.0 | 50.0 |
| 6678 | camera02 | 42.1 | 26.7 | 61.5 | 6.5 | 38.5 | 16.0 | 20.0 | 17.4 |
| 6678 | camera03 | 100.0 | 13.8 | 55.6 | 52.6 | 41.7 | 16.3 | 47.6 | 22.2 |
| 6678 | camera04 | 45.5 | 55.6 | 71.4 | 34.8 | 71.4 | 15.7 | 37.0 | 13.1 |
| 6678 | camera05 | 13.8 | 45.5 | 38.5 | 37.0 | 100.0 | 13.8 | 18.2 | 18.6 |
| 6678 | top-3 per frame (Δ=0) | 83.3 | 71.4 | 71.4 | 45.5 | 71.4 | 57.1 | 27.0 | 50.0 |
| 6678 | top-3 ±5 s window (Δ=5) | 100.0 | 100.0 | 83.3 | 83.3 | 83.3 | 57.1 | 66.7 | 66.7 |
| 6678 | top-3 per video (Δ=∞) | 100.0 | 38.5 | 62.5 | 71.4 | 71.4 | 21.1 | 21.7 | 30.0 |
| 7678 | camera01 | 83.3 | 11.8 | 40.0 | 55.6 | 62.5 | 22.9 | 55.6 | 53.3 |
| 7678 | camera02 | 25.0 | 9.1 | 47.1 | 13.8 | 21.1 | 6.0 | 11.8 | 22.2 |
| 7678 | camera03 | 83.3 | 45.5 | 38.5 | 30.8 | 45.5 | 30.8 | 40.0 | 34.5 |
| 7678 | camera04 | 32.3 | 71.4 | 71.4 | 9.7 | 83.3 | 21.6 | 52.6 | 21.6 |
| 7678 | camera05 | 12.9 | 38.5 | 45.5 | 12.5 | 35.7 | 38.5 | 58.8 | 18.6 |
| 7678 | top-3 per frame (Δ=0) | 83.3 | 55.6 | 35.7 | 19.4 | 38.5 | 12.5 | 83.3 | 42.1 |
| 7678 | top-3 ±5 s window (Δ=5) | 100.0 | 83.3 | 71.4 | 53.3 | 71.4 | 21.1 | 100.0 | 90.9 |
| 7678 | top-3 per video (Δ=∞) | 83.3 | 83.3 | 62.5 | 22.9 | 41.7 | 25.0 | 71.4 | 43.5 |

**In-domain (Exp. A), number of predicted segments (ground truth: 5)**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 43 | 35 | 9 | 5 | 17 | 22 | 13 | 16 |
| 5678 | camera02 | 54 | 52 | 17 | 28 | 37 | 21 | 12 | 16 |
| 5678 | camera03 | 47 | 31 | 19 | 35 | 49 | 29 | 14 | 42 |
| 5678 | camera04 | 18 | 23 | 9 | 26 | 21 | 47 | 32 | 18 |
| 5678 | camera05 | 53 | 13 | 13 | 23 | 18 | 35 | 44 | 11 |
| 5678 | top-3 per frame (Δ=0) | 9 | 9 | 7 | 15 | 13 | 57 | 22 | 12 |
| 5678 | top-3 ±5 s window (Δ=5) | 7 | 5 | 7 | 7 | 7 | 27 | 6 | 8 |
| 5678 | top-3 per video (Δ=∞) | 24 | 7 | 9 | 29 | 9 | 41 | 30 | 41 |
| 6678 | camera01 | 9 | 49 | 9 | 6 | 7 | 17 | 5 | 11 |
| 6678 | camera02 | 14 | 25 | 8 | 57 | 21 | 20 | 35 | 18 |
| 6678 | camera03 | 5 | 53 | 13 | 14 | 19 | 44 | 16 | 40 |
| 6678 | camera04 | 17 | 13 | 9 | 18 | 9 | 46 | 22 | 56 |
| 6678 | camera05 | 53 | 17 | 21 | 22 | 5 | 53 | 39 | 38 |
| 6678 | top-3 per frame (Δ=0) | 7 | 9 | 9 | 17 | 9 | 9 | 32 | 11 |
| 6678 | top-3 ±5 s window (Δ=5) | 5 | 5 | 7 | 7 | 7 | 9 | 10 | 7 |
| 6678 | top-3 per video (Δ=∞) | 5 | 21 | 11 | 9 | 9 | 33 | 41 | 15 |
| 7678 | camera01 | 7 | 46 | 20 | 13 | 11 | 30 | 13 | 10 |
| 7678 | camera02 | 27 | 61 | 12 | 24 | 33 | 62 | 63 | 13 |
| 7678 | camera03 | 7 | 17 | 21 | 21 | 17 | 21 | 20 | 24 |
| 7678 | camera04 | 26 | 9 | 9 | 57 | 7 | 32 | 14 | 32 |
| 7678 | camera05 | 57 | 21 | 17 | 59 | 23 | 21 | 12 | 38 |
| 7678 | top-3 per frame (Δ=0) | 7 | 13 | 23 | 26 | 21 | 59 | 7 | 14 |
| 7678 | top-3 ±5 s window (Δ=5) | 5 | 7 | 9 | 10 | 9 | 33 | 5 | 6 |
| 7678 | top-3 per video (Δ=∞) | 7 | 7 | 11 | 30 | 19 | 27 | 9 | 18 |

**OOD zero-shot (Exp. B), frame accuracy**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 0.903 | 0.940 | 0.834 | 0.911 | 0.841 | 0.477 | 0.569 | 0.879 |
| 5678 | camera02 | 0.951 | 0.909 | 0.761 | 0.855 | 0.910 | 0.593 | 0.530 | 0.932 |
| 5678 | camera03 | 0.934 | 0.974 | 0.904 | 0.915 | 0.810 | 0.958 | 0.945 | 0.921 |
| 5678 | camera04 | 0.730 | 0.964 | 0.828 | 0.920 | 0.822 | 0.779 | 0.905 | 0.830 |
| 5678 | camera05 | 0.828 | 0.884 | 0.842 | 0.844 | 0.879 | 0.900 | 0.919 | 0.909 |
| 5678 | top-3 per frame (Δ=0) | 0.868 | 0.944 | 0.882 | 0.945 | 0.877 | 0.819 | 0.777 | 0.907 |
| 5678 | top-3 ±5 s window (Δ=5) | 0.868 | 0.944 | 0.882 | 0.940 | 0.875 | 0.821 | 0.773 | 0.907 |
| 5678 | top-3 per video (Δ=∞) | 0.964 | 0.947 | 0.888 | 0.954 | 0.862 | 0.661 | 0.958 | 0.917 |
| 6678 | camera01 | 0.955 | 0.891 | 0.912 | 0.921 | 0.784 | 0.510 | 0.573 | 0.784 |
| 6678 | camera02 | 0.922 | 0.922 | 0.661 | 0.895 | 0.901 | 0.600 | 0.527 | 0.900 |
| 6678 | camera03 | 0.928 | 0.969 | 0.923 | 0.876 | 0.867 | 0.898 | 0.952 | 0.903 |
| 6678 | camera04 | 0.870 | 0.906 | 0.876 | 0.961 | 0.841 | 0.698 | 0.922 | 0.815 |
| 6678 | camera05 | 0.840 | 0.877 | 0.862 | 0.831 | 0.891 | 0.910 | 0.902 | 0.867 |
| 6678 | top-3 per frame (Δ=0) | 0.952 | 0.953 | 0.920 | 0.952 | 0.958 | 0.796 | 0.885 | 0.812 |
| 6678 | top-3 ±5 s window (Δ=5) | 0.953 | 0.951 | 0.928 | 0.953 | 0.959 | 0.799 | 0.885 | 0.810 |
| 6678 | top-3 per video (Δ=∞) | 0.962 | 0.972 | 0.921 | 0.953 | 0.936 | 0.860 | 0.935 | 0.842 |
| 7678 | camera01 | 0.902 | 0.847 | 0.852 | 0.809 | 0.939 | 0.546 | 0.580 | 0.816 |
| 7678 | camera02 | 0.930 | 0.917 | 0.609 | 0.803 | 0.892 | 0.693 | 0.500 | 0.923 |
| 7678 | camera03 | 0.940 | 0.974 | 0.903 | 0.863 | 0.897 | 0.974 | 0.926 | 0.923 |
| 7678 | camera04 | 0.851 | 0.894 | 0.816 | 0.909 | 0.855 | 0.803 | 0.944 | 0.815 |
| 7678 | camera05 | 0.784 | 0.817 | 0.807 | 0.791 | 0.940 | 0.908 | 0.967 | 0.897 |
| 7678 | top-3 per frame (Δ=0) | 0.919 | 0.959 | 0.950 | 0.835 | 0.968 | 0.878 | 0.965 | 0.922 |
| 7678 | top-3 ±5 s window (Δ=5) | 0.924 | 0.960 | 0.949 | 0.837 | 0.971 | 0.880 | 0.965 | 0.922 |
| 7678 | top-3 per video (Δ=∞) | 0.964 | 0.975 | 0.959 | 0.933 | 0.960 | 0.854 | 0.955 | 0.918 |

**OOD zero-shot (Exp. B), Edit score**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 6.4 | 22.7 | 25.0 | 23.8 | 7.7 | 20.0 | 16.7 | 20.8 |
| 5678 | camera02 | 35.7 | 27.8 | 7.6 | 10.6 | 14.3 | 45.5 | 13.8 | 11.6 |
| 5678 | camera03 | 22.7 | 26.3 | 26.3 | 41.7 | 4.5 | 12.2 | 18.5 | 62.5 |
| 5678 | camera04 | 4.8 | 15.2 | 12.8 | 14.7 | 7.9 | 5.7 | 7.7 | 14.3 |
| 5678 | camera05 | 27.8 | 16.7 | 13.9 | 26.3 | 10.6 | 5.6 | 41.7 | 17.9 |
| 5678 | top-3 per frame (Δ=0) | 10.0 | 50.0 | 15.2 | 13.2 | 9.3 | 19.2 | 4.8 | 62.5 |
| 5678 | top-3 ±5 s window (Δ=5) | 25.0 | 83.3 | 41.7 | 31.2 | 16.7 | 50.0 | 15.6 | 83.3 |
| 5678 | top-3 per video (Δ=∞) | 62.5 | 50.0 | 22.7 | 25.0 | 12.2 | 19.2 | 27.8 | 50.0 |
| 6678 | camera01 | 27.8 | 22.7 | 10.9 | 33.3 | 10.6 | 12.8 | 22.7 | 11.6 |
| 6678 | camera02 | 29.4 | 21.7 | 7.0 | 25.0 | 14.3 | 4.1 | 40.0 | 6.9 |
| 6678 | camera03 | 19.2 | 55.6 | 16.1 | 13.5 | 13.9 | 35.7 | 15.2 | 27.8 |
| 6678 | camera04 | 9.8 | 20.0 | 18.5 | 15.2 | 8.8 | 10.0 | 9.8 | 23.8 |
| 6678 | camera05 | 23.8 | 12.5 | 15.6 | 13.2 | 11.9 | 7.1 | 13.2 | 10.6 |
| 6678 | top-3 per frame (Δ=0) | 19.2 | 13.9 | 13.2 | 20.8 | 29.4 | 15.6 | 15.2 | 21.7 |
| 6678 | top-3 ±5 s window (Δ=5) | 20.8 | 21.7 | 35.7 | 31.2 | 55.6 | 20.8 | 45.5 | 45.5 |
| 6678 | top-3 per video (Δ=∞) | 41.7 | 100.0 | 25.0 | 27.8 | 11.1 | 10.2 | 23.8 | 16.1 |
| 7678 | camera01 | 19.2 | 9.1 | 19.2 | 7.0 | 11.1 | 23.8 | 35.7 | 12.2 |
| 7678 | camera02 | 11.6 | 15.2 | 7.6 | 7.4 | 9.4 | 8.5 | 25.0 | 13.9 |
| 7678 | camera03 | 31.2 | 38.5 | 16.1 | 13.9 | 11.6 | 71.4 | 16.7 | 25.0 |
| 7678 | camera04 | 11.9 | 14.7 | 9.1 | 8.6 | 12.5 | 10.6 | 10.4 | 13.9 |
| 7678 | camera05 | 55.6 | 15.2 | 6.1 | 15.6 | 9.6 | 8.8 | 50.0 | 10.6 |
| 7678 | top-3 per frame (Δ=0) | 16.7 | 27.8 | 41.7 | 13.2 | 15.6 | 8.3 | 71.4 | 27.8 |
| 7678 | top-3 ±5 s window (Δ=5) | 31.2 | 35.7 | 83.3 | 41.7 | 26.3 | 31.2 | 100.0 | 62.5 |
| 7678 | top-3 per video (Δ=∞) | 22.7 | 71.4 | 45.5 | 8.9 | 27.8 | 7.9 | 45.5 | 14.3 |

**OOD zero-shot (Exp. B), F1@50**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 9.6 | 29.6 | 32.0 | 30.8 | 11.4 | 13.3 | 22.9 | 27.6 |
| 5678 | camera02 | 52.6 | 43.5 | 8.5 | 15.4 | 15.0 | 37.5 | 17.6 | 20.8 |
| 5678 | camera03 | 37.0 | 41.7 | 41.7 | 58.8 | 5.2 | 21.7 | 31.3 | 76.9 |
| 5678 | camera04 | 7.3 | 26.3 | 18.2 | 20.5 | 11.8 | 6.5 | 11.4 | 25.0 |
| 5678 | camera05 | 34.8 | 28.6 | 19.5 | 33.3 | 15.4 | 10.5 | 58.8 | 24.2 |
| 5678 | top-3 per frame (Δ=0) | 14.5 | 66.7 | 26.3 | 23.3 | 13.6 | 19.4 | 7.3 | 76.9 |
| 5678 | top-3 ±5 s window (Δ=5) | 32.0 | 90.9 | 58.8 | 47.6 | 22.9 | 40.0 | 21.6 | 90.9 |
| 5678 | top-3 per video (Δ=∞) | 76.9 | 66.7 | 37.0 | 40.0 | 17.4 | 19.4 | 43.5 | 66.7 |
| 6678 | camera01 | 43.5 | 29.6 | 19.6 | 50.0 | 15.4 | 9.1 | 22.2 | 16.7 |
| 6678 | camera02 | 45.5 | 35.7 | 7.9 | 32.0 | 20.0 | 4.8 | 40.0 | 13.0 |
| 6678 | camera03 | 32.3 | 71.4 | 27.8 | 23.8 | 19.5 | 52.6 | 26.3 | 43.5 |
| 6678 | camera04 | 14.3 | 33.3 | 31.3 | 26.3 | 12.9 | 14.5 | 17.9 | 38.5 |
| 6678 | camera05 | 30.8 | 17.8 | 21.6 | 14.0 | 17.0 | 10.7 | 23.3 | 19.2 |
| 6678 | top-3 per frame (Δ=0) | 32.3 | 24.4 | 23.3 | 34.5 | 45.5 | 21.6 | 21.1 | 28.6 |
| 6678 | top-3 ±5 s window (Δ=5) | 34.5 | 35.7 | 52.6 | 47.6 | 71.4 | 27.6 | 62.5 | 50.0 |
| 6678 | top-3 per video (Δ=∞) | 58.8 | 100.0 | 40.0 | 43.5 | 16.0 | 14.8 | 38.5 | 27.8 |
| 7678 | camera01 | 32.3 | 10.0 | 32.3 | 10.5 | 20.0 | 15.4 | 31.6 | 17.4 |
| 7678 | camera02 | 20.8 | 26.3 | 8.5 | 11.0 | 13.8 | 12.5 | 28.6 | 19.5 |
| 7678 | camera03 | 47.6 | 55.6 | 27.8 | 19.5 | 12.5 | 83.3 | 28.6 | 40.0 |
| 7678 | camera04 | 17.0 | 25.6 | 13.3 | 12.7 | 17.8 | 15.4 | 18.9 | 19.5 |
| 7678 | camera05 | 57.1 | 15.8 | 9.2 | 16.2 | 14.0 | 16.1 | 66.7 | 15.4 |
| 7678 | top-3 per frame (Δ=0) | 28.6 | 43.5 | 58.8 | 18.6 | 27.0 | 9.2 | 83.3 | 43.5 |
| 7678 | top-3 ±5 s window (Δ=5) | 47.6 | 52.6 | 90.9 | 47.1 | 41.7 | 28.6 | 100.0 | 76.9 |
| 7678 | top-3 per video (Δ=∞) | 37.0 | 83.3 | 62.5 | 13.1 | 43.5 | 11.8 | 62.5 | 25.0 |

**OOD zero-shot (Exp. B), number of predicted segments (ground truth: 5)**

| Seed | Method | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 78 | 22 | 20 | 21 | 65 | 25 | 30 | 24 |
| 5678 | camera02 | 14 | 18 | 66 | 47 | 35 | 11 | 29 | 43 |
| 5678 | camera03 | 22 | 19 | 19 | 12 | 111 | 41 | 27 | 8 |
| 5678 | camera04 | 105 | 33 | 39 | 34 | 63 | 87 | 65 | 35 |
| 5678 | camera05 | 18 | 30 | 36 | 19 | 47 | 90 | 12 | 28 |
| 5678 | top-3 per frame (Δ=0) | 50 | 10 | 33 | 38 | 54 | 26 | 104 | 8 |
| 5678 | top-3 ±5 s window (Δ=5) | 20 | 6 | 12 | 16 | 30 | 10 | 32 | 6 |
| 5678 | top-3 per video (Δ=∞) | 8 | 10 | 22 | 20 | 41 | 26 | 18 | 10 |
| 6678 | camera01 | 18 | 22 | 46 | 15 | 47 | 39 | 22 | 43 |
| 6678 | camera02 | 17 | 23 | 71 | 20 | 35 | 121 | 10 | 72 |
| 6678 | camera03 | 26 | 9 | 31 | 37 | 36 | 14 | 33 | 18 |
| 6678 | camera04 | 51 | 25 | 27 | 33 | 57 | 50 | 51 | 21 |
| 6678 | camera05 | 21 | 40 | 32 | 38 | 42 | 70 | 38 | 47 |
| 6678 | top-3 per frame (Δ=0) | 26 | 36 | 38 | 24 | 17 | 32 | 33 | 23 |
| 6678 | top-3 ±5 s window (Δ=5) | 24 | 23 | 14 | 16 | 9 | 24 | 11 | 11 |
| 6678 | top-3 per video (Δ=∞) | 12 | 5 | 20 | 18 | 45 | 49 | 21 | 31 |
| 7678 | camera01 | 26 | 55 | 26 | 71 | 45 | 21 | 14 | 41 |
| 7678 | camera02 | 43 | 33 | 66 | 68 | 53 | 59 | 16 | 36 |
| 7678 | camera03 | 16 | 13 | 31 | 36 | 43 | 7 | 30 | 20 |
| 7678 | camera04 | 42 | 34 | 55 | 58 | 40 | 47 | 48 | 36 |
| 7678 | camera05 | 9 | 33 | 82 | 32 | 52 | 57 | 10 | 47 |
| 7678 | top-3 per frame (Δ=0) | 30 | 18 | 12 | 38 | 32 | 60 | 7 | 18 |
| 7678 | top-3 ±5 s window (Δ=5) | 16 | 14 | 6 | 12 | 19 | 16 | 5 | 8 |
| 7678 | top-3 per video (Δ=∞) | 22 | 7 | 11 | 56 | 18 | 63 | 11 | 35 |

</details>

<details>
<summary><b>Stage-2 best validation accuracy per fold (collapse check)</b></summary>

| Seed | Camera | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | fold 6 | fold 7 |
|---|---|---|---|---|---|---|---|---|---|
| 5678 | camera01 | 0.57 | 0.80 | 0.71 | 0.66 | 0.73 | 0.53 | 0.63 | 0.74 |
| 5678 | camera02 | 0.44 | 0.74 | 0.71 | 0.62 | 0.69 | 0.53 | 0.63 | 0.71 |
| 5678 | camera03 | 0.56 | 0.81 | 0.67 | 0.62 | 0.75 | 0.53 | 0.69 | 0.72 |
| 5678 | camera04 | 0.58 | 0.78 | 0.73 | 0.66 | 0.70 | 0.58 | 0.69 | 0.77 |
| 5678 | camera05 | 0.58 | 0.79 | 0.73 | 0.65 | 0.70 | 0.54 | 0.67 | 0.76 |
| 6678 | camera01 | 0.72 | 0.80 | 0.56 | 0.66 | 0.79 | 0.70 | 0.42 | 0.70 |
| 6678 | camera02 | 0.64 | 0.73 | 0.55 | 0.66 | 0.75 | 0.69 | 0.47 | 0.57 |
| 6678 | camera03 | 0.72 | 0.82 | 0.55 | 0.65 | 0.80 | 0.71 | 0.54 | 0.67 |
| 6678 | camera04 | 0.72 | 0.78 | 0.56 | 0.68 | 0.77 | 0.73 | 0.58 | 0.67 |
| 6678 | camera05 | 0.71 | 0.79 | 0.57 | 0.66 | 0.80 | 0.70 | 0.49 | 0.70 |
| 7678 | camera01 | 0.75 | 0.80 | 0.61 | 0.70 | 0.72 | 0.55 | 0.68 | 0.65 |
| 7678 | camera02 | 0.66 | 0.76 | 0.58 | 0.71 | 0.69 | 0.53 | 0.72 | 0.59 |
| 7678 | camera03 | 0.77 | 0.81 | 0.57 | 0.67 | 0.72 | 0.55 | 0.82 | 0.62 |
| 7678 | camera04 | 0.76 | 0.77 | 0.63 | 0.71 | 0.70 | 0.58 | 0.85 | 0.65 |
| 7678 | camera05 | 0.77 | 0.80 | 0.64 | 0.70 | 0.74 | 0.56 | 0.79 | 0.67 |

Minimum over all 120 folds: 0.42.

</details>

<details>
<summary><b>Previous release: constant-rate labels (superseded)</b></summary>

First release: labels mapped with the constant-rate estimate (≈ 7.73 fps), every 8th frame, Stage 2 seed 5678 re-fitted on Stage-1 features of an earlier run. The last column is the change in frame accuracy from these numbers to the new three-seed mean.

**In-domain (Exp. A)**

| Method | Mean fold acc. ± SD | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 | Δ acc. vs. new mean |
|---|---|---|---|---|---|---|---|---|
| camera01 | 0.874 ± 0.090 | 0.869 | 0.857 | 47.8 | 58.9 | 56.6 | 55.9 | -0.026 |
| camera02 | 0.735 ± 0.158 | 0.740 | 0.691 | 25.3 | 32.3 | 29.2 | 28.6 | -0.024 |
| camera03 | 0.899 ± 0.051 | 0.900 | 0.889 | 39.6 | 54.1 | 54.1 | 53.5 | -0.003 |
| camera04 | 0.868 ± 0.108 | 0.870 | 0.854 | 38.0 | 50.0 | 48.5 | 48.5 | +0.004 |
| camera05 | 0.892 ± 0.105 | 0.888 | 0.877 | 44.6 | 57.8 | 56.9 | 54.8 | +0.028 |
| top-3 per frame (Δ=0) | 0.891 ± 0.070 | 0.891 | 0.880 | 50.4 | 63.9 | 62.1 | 62.1 | -0.016 |
| top-3 ±5 s window (Δ=5) | 0.891 ± 0.071 | 0.891 | 0.881 | 75.5 | 84.5 | 82.4 | 82.4 | -0.017 |
| top-3 per video (Δ=∞) | 0.900 ± 0.080 | 0.898 | 0.889 | 66.5 | 75.8 | 74.8 | 74.8 | +0.003 |

**OOD zero-shot (Exp. B)**

| Method | Mean fold acc. ± SD | Frame acc. | Macro F1 | Edit | F1@10 | F1@25 | F1@50 | Δ acc. vs. new mean |
|---|---|---|---|---|---|---|---|---|
| camera01 | 0.936 ± 0.032 | 0.936 | 0.916 | 23.0 | 36.0 | 36.0 | 33.1 | -0.146 |
| camera02 | 0.939 ± 0.031 | 0.939 | 0.909 | 36.0 | 48.4 | 48.4 | 47.7 | -0.146 |
| camera03 | 0.965 ± 0.018 | 0.965 | 0.956 | 66.4 | 76.2 | 76.2 | 76.2 | -0.045 |
| camera04 | 0.902 ± 0.054 | 0.902 | 0.869 | 12.4 | 20.9 | 20.9 | 19.1 | -0.045 |
| camera05 | 0.886 ± 0.080 | 0.886 | 0.863 | 20.6 | 32.2 | 31.7 | 30.6 | -0.015 |
| top-3 per frame (Δ=0) | 0.956 ± 0.026 | 0.956 | 0.944 | 36.9 | 51.5 | 51.5 | 51.5 | -0.054 |
| top-3 ±5 s window (Δ=5) | 0.956 ± 0.027 | 0.956 | 0.945 | 61.0 | 74.3 | 74.3 | 74.3 | -0.054 |
| top-3 per video (Δ=∞) | 0.965 ± 0.020 | 0.965 | 0.956 | 57.0 | 69.8 | 69.8 | 69.8 | -0.046 |

</details>

### Labels and frame timing

The exports run at **7.48 fps** (one capture timestamp per frame in
`export_full/tables_timestamps.arrow`, identical for all five cameras), and the recordings continue
50–80 s after the last annotated second. The first release estimated the frame rate as
`n_frames / last annotated second` (≈ 7.73 fps), which placed every phase boundary late, by up to
~35 s at the start of Microsurgery. `vitrectomy/build_dataframe.py` now assigns each frame the label
at its real capture time and samples the frame nearest to each integer second (true 1 Hz; the
first release took every 8th frame, ≈ 1.07 s). Example, Run 3: Microsurgery starts at 1107 s in the
annotation; the old mapping put it at ≈ 1144 s, the new one at 1108 s (first sample ≥ 1107 s).

Dataset after the timestamp-exact build (identical for all five cameras; samples at 1 Hz;
-1 = before the first / after the last annotated second):

| Run | video_idx | fps (timestamps) | Recording [s] | Annotated [s] | Samples | Room_Prep | Patient_Entry | Sterile_Draping | Microsurgery | Emergence | Unlabeled | Labeled | Labeled, old build | Samples, old build |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 7.484 | 2212.8 | 116–2143 | 2213 | 214 | 527 | 524 | 527 | 236 | 185 | 2028 | 1959 | 2071 |
| 2 | 1 | 7.484 | 2179.8 | 142–2109 | 2180 | 214 | 406 | 417 | 688 | 242 | 213 | 1967 | 1902 | 2040 |
| 3 | 2 | 7.483 | 2029.0 | 127–1964 | 2030 | 227 | 363 | 391 | 601 | 255 | 193 | 1837 | 1775 | 1898 |
| 4 | 3 | 7.486 | 1926.6 | 105–1854 | 1927 | 220 | 343 | 403 | 567 | 216 | 178 | 1749 | 1700 | 1803 |
| 5 | 4 | 7.484 | 2513.1 | 82–2445 | 2514 | 230 | 360 | 708 | 759 | 307 | 150 | 2364 | 2273 | 2352 |
| 6 | 5 | 7.482 | 2513.0 | 167–2443 | 2514 | 243 | 307 | 645 | 797 | 284 | 238 | 2276 | 2190 | 2351 |
| 7 | 6 | 7.485 | 2036.9 | 95–1986 | 2037 | 206 | 282 | 461 | 718 | 224 | 146 | 1891 | 1814 | 1906 |
| 8 | 7 | 7.484 | 2573.0 | 88–2492 | 2574 | 198 | 318 | 497 | 1134 | 258 | 169 | 2405 | 2323 | 2408 |
| 9 (OOD) | 8 | 7.485 | 2764.2 | 128–2698 | 2765 | 191 | 301 | 569 | 1104 | 404 | 196 | 2569 | 2464 | 2587 |
| **Total** | | | | | **20754** | **1943** | **3207** | **4615** | **6895** | **2426** | **1668** | **19086** | **18400** | **19416** |

### Compute

One NVIDIA RTX PRO 6000 (Blackwell) per process. One fold of `run_expAB.sh` takes about 7 min
(Stage 1 incl. feature export ≈ 6.5 min, Stage 2 + evaluation ≈ 30 s). The full Experiment A + B
(5 cameras × 8 folds) took 2 h 41 min with two processes on two GPUs (cameras 01–02 and 03–05).
One extra Stage-2 seed over all 40 folds takes about 15 min on one GPU (≈ 20 s per fold).

## Pipeline

```
raw frames ──► build_dataframe ──► Stage 1: ResNet-50 ──► 2048-d features ──► Stage 2: MS-TCN ──► fold<k>_preds.pkl ──► voting / metrics
               (1 fps, labels)     (per frame)           (per video)          (causal, per video)   (y_true, y_pred, y_soft)
```

Training protocol:

| | Stage 1 (ResNet-50) | Stage 2 (MS-TCN) |
|---|---|---|
| Config | `modules/cnn/config/config_feature_extract_vitrectomy.yml` | `modules/mstcn/config/config_tcn_vitrectomy.yml` |
| Seed | 1234 + fold | 5678 + fold |
| Early-stopping patience | 3 (default) | 30 |
| Epochs | 5 – 15 | 10 – 30 |
| Learning rate | 1e-4 | 7e-5 |

Stage 1 and Stage 2 use different seeds. `predictions/camera0*` come from one end-to-end
`run_expAB.sh` run with exactly these settings (both stages trained, all 8 folds × 5 cameras).
`predictions/seed6678/` and `predictions/seed7678/` re-train only Stage 2 on the same Stage-1
features with base seeds 6678 and 7678 (`run_stage2_refit.sh`, see [Run](#run-training-from-raw-frames)).

### Setup

```bash
bash setup_env.sh              # creates venv_tecno/ and installs requirements.txt
source venv_tecno/bin/activate
```

### Data layout

The frames are not part of this repository. They are in the Bifocal-OR dataset on
[Hugging Face](https://huggingface.co/datasets/AngeloHenriques/Bifocal-OR) (gated access,
CC BY-NC 4.0). The phase annotations are included here (`annotations/`). The code expects the
layout of the raw capture:

```
data/raw/rec_*_Run_<n>/export_full/color/color_<frame>_<camera>.jpg
data/raw/rec_*_Run_<n>/export_full/tables_timestamps.arrow   # capture time of every frame
annotations/annotations_<n>.csv         # phase_id,phase_name,start_sec,end_sec
```

The dataset stores the RGB frames of each run and room camera as one `.tar`
(`runs/runNN/room/color/cameraXX.tar`) and the frame timestamps as
`runs/runNN/room/timestamps.arrow`. This unpacks a local copy of the dataset into the layout
above. The folder names are the ones in the run list of `vitrectomy/build_dataframe.py`:

```bash
HF=/path/to/Bifocal-OR    # local copy of the Hugging Face dataset
RUNS=(rec_12-03-2026_11-25_Run_1 rec_12-03-2026_12-40_Run_2 rec_12-03-2026_14-02_Run_3
      rec_12-03-2026_14-52_Run_4 rec_13-03-2026_09-32_Run_5 rec_13-03-2026_10-42_Run_6
      rec_13-03-2026_13-26_Run_7 rec_13-03-2026_14-19_Run_8 rec_13-03-2026_15-37_Run_9)
for i in "${!RUNS[@]}"; do
  n=$(printf %02d $((i + 1)))
  dst=data/raw/${RUNS[$i]}/export_full
  mkdir -p "$dst/color"
  for c in 01 02 03 04 05; do
    tar -xf "$HF/runs/run$n/room/color/camera$c.tar" -C "$dst/color"
  done
  cp "$HF/runs/run$n/room/timestamps.arrow" "$dst/tables_timestamps.arrow"
done
```

Camera and phase names are defined in a single place, `utils/vitrectomy_constants.py`.
To adapt the pipeline to another multi-view dataset, edit that file and the run list in
`vitrectomy/build_dataframe.py`.

### Run (training from raw frames)

```bash
bash build_all_dataframes.sh        # -> data/vitrectomy/dataframes/vitrectomy_1fps_<camera>.pkl
bash run_expAB.sh                   # all cameras × 8 folds, in-domain + OOD (resumable)
bash run_expAB.sh camera03 -- 0 3   # camera / fold subset
```

`run_stage2_refit.sh` re-trains only Stage 2 on the Stage-1 features of an earlier
`run_expAB.sh` run (found via its `fold<k>_stage1.log`) and writes the same pickles. `SEED`
overrides the Stage-2 base seed on the command line, so the config file is never edited and
several seeds can run at the same time. The shipped predictions were produced with:

```bash
bash build_all_dataframes.sh                       # needs pyarrow (requirements.txt)
OUTPUTS=outputs_exact bash run_expAB.sh            # Stage 1 + Stage 2, seed 5678
for S in 6678 7678; do                             # two extra Stage-2 seeds
  SEED=$S OUTPUTS=outputs_exact REFIT_DIR=outputs_exact/seed$S bash run_stage2_refit.sh
done
```

With seed 5678 on the same features, `run_stage2_refit.sh` reproduces the `run_expAB.sh`
predictions (checked on camera03 / fold 0: identical labels, max. softmax difference 4e-4).

`run_loo_cv.sh` runs Stage 1 → Stage 2 → evaluation for a single camera. It passes the
Stage 1 feature directory to Stage 2 automatically. Results go to
`outputs/expAB_loo_cv_<camera>/`.

### Analysis

Works directly on the shipped predictions, no GPU and no data needed. Use
`--folds 0 1 2 3 4 5 6 7` (Experiment A has 8 folds; the default is 9).

```bash
F="--root predictions --pattern camera* --folds 0 1 2 3 4 5 6 7"

python evaluation/loo_voting.py compare-cameras --root predictions --pattern 'camera*'
python evaluation/loo_voting.py dynamic  $F --windows 0 5 inf --out results
python evaluation/loo_voting.py pf-vs-pv $F
python evaluation/loo_voting.py dropout  $F
# add --ood to dynamic / pf-vs-pv / dropout for Experiment B
# other Stage-2 seeds: --root predictions/seed6678  or  --root predictions/seed7678

python evaluation/loo_aggregate.py --results_dir predictions/camera03        # one camera, pooled metrics
python evaluation/loo_aggregate.py --results_dir predictions/camera03 --ood
```

`pf-vs-pv` and `dropout` write their reports into `--root`, i.e. into `predictions/`.
Point `--root` to a copy if you want to keep that folder untouched.

`static` (hard vs. soft voting over all camera subsets) reads the single-camera
`run_loo_cv.sh` layout (`outputs/loo_cv_<camera>/`), not `predictions/`.

## Repository structure

```
train.py                 entry point; model/dataset/module classes are loaded from the YAML config
models/                  ResNet-50 encoder, multi-stage TCN
modules/                 Lightning modules + configs for Stage 1 and Stage 2
datasets/                vitrectomy datasets and LOO split logic
vitrectomy/              dataframe builder (frames + annotations -> 1 fps labels)
annotations/             phase annotations, Runs 1-9 (start/end second per phase)
evaluation/              inference, aggregation, segmental metrics, camera ensembling
utils/                   config parsing, metrics, dataset constants
predictions/             per-frame predictions of the final models, 3 Stage-2 seeds (see below)
run_expAB.sh             Experiment A + B for all cameras
run_loo_cv.sh            Stage 1 -> Stage 2 -> evaluation for one camera
run_stage2_refit.sh      Stage 2 only, on existing Stage-1 features
build_all_dataframes.sh  frames + annotations -> dataframes
```

### `predictions/`

```
predictions/camera0{1..5}/
    fold{0..7}_preds.pkl       Experiment A: held-out run of fold k (fold k = Run k+1)
    fold{0..7}_preds_ood.pkl   Experiment B: the model of fold k applied to Run 9
    loo_summary.txt            pooled metrics + confusion matrix, Experiment A
    loo_summary_ood.txt        same for Experiment B
predictions/seed{6678,7678}/camera0{1..5}/
    same files, Stage 2 re-trained with base seed 6678 / 7678 on the same Stage-1 features
```

Each pickle is a dict with `test_fold`, `y_true` `[N]` (labels 0–4, labeled frames only),
`y_pred` `[N]` (argmax of the last MS-TCN stage) and `y_soft` `[N, 5]` (softmax per frame).
Frames are at 1 fps. Phase indices: 0 Room_Prep, 1 Patient_Entry, 2 Sterile_Draping,
3 Microsurgery, 4 Emergence.

```python
import pickle
d = pickle.load(open("predictions/camera03/fold0_preds.pkl", "rb"))
d["y_true"], d["y_pred"], d["y_soft"]
```

## Acknowledgements

Built on [TeCNO](https://github.com/tobiascz/TeCNO) by Czempiel et al. and distributed
under the same GPL-3.0 license. If you use this code, please cite the original work:

```bibtex
@inproceedings{czempiel2020,
  author    = {Tobias Czempiel and Magdalini Paschali and Matthias Keicher and
               Walter Simson and Hubertus Feussner and Seong Tae Kim and Nassir Navab},
  title     = {TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional Networks},
  booktitle = {MICCAI 2020},
  series    = {LNCS},
  volume    = {12263},
  pages     = {343--352},
  publisher = {Springer},
  year      = {2020},
}
```
