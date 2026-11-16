"""Bifocal-OR viewer: all streams of a run at one moment, Figure-2 style.

Renders the main room camera, the four other room cameras, the microscope, the phase
timeline, the speaker activity and the ambient waveform for one timestamp (PNG), a time
range (MP4), or sheets that check the synchronization.

Examples (Windows PowerShell):
    python bifocal_viewer.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --t 1300
    python bifocal_viewer.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --clip 1290 1320
    python bifocal_viewer.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --clap start
    python bifocal_viewer.py --root "E:\\Ophthalmo_4D" --labels .\\Bifocal-OR --run 3 --pair 1300 --micro-offset 12.4

Modes
    --t T            one PNG at time T (seconds on the room-camera timeline)
    --clip T0 T1     MP4 from T0 to T1 (--fps frames per second, default 5)
    --clap start|end audio-video check: finds the clap in the ambient audio and shows
                     the room-camera frames around it. If the hands meet at 0.0 s, the
                     audio offset is right; otherwise the sheet shows by how much it is off.
    --pair T         room camera and microscope side by side from T-2 s to T+2 s, to judge
                     a microscope offset (use --micro-offset to try values)
    --scan T0 T1     contact sheet of the main camera every --step seconds (default 2), e.g. to
                     find the clap at the start of a run

Where things are found (override with the options in brackets)
    run folder        ROOT/rec_*_Run_<n>
    room frames       <run folder>/export_full/color/color_<frame>_camera0X.jpg
                      (falls back to camera0X*.mkv in the run folder)
    microscope        <run folder>/*Manual*/video_h265.mp4 (Run 1 has two parts) [--micro]
    audio             ROOT/*DVR*/Run_<n>/*.wav [--audio-dir]
    phases, speakers  LABELS/annotations/... (release layout) or annotations_<n>.csv [--labels]
    frame times       LABELS/sync/room_frame_times/runNN.csv, else export_full/tables_timestamps.arrow
                      (pyarrow), else a constant 7.484 frames/s [--video-fps]
    microscope times  LABELS/sync/microscope_frame_times/runNN[_partK].csv: real time of the
                      microscope frames (the video file itself runs too fast)
    offsets           LABELS/sync/offsets.csv; offset_s = time on the room-camera timeline
                      at which the stream (audio, microscope clock) starts: stream time = t - offset_s
                      [--audio-offset, --micro-offset; --micro-rate only without frame-time files]

Needs: numpy, matplotlib, pillow. Optional: opencv-python (microscope, MKV, MP4 output),
scipy (32-bit float WAV files).
    pip install numpy matplotlib pillow opencv-python scipy
"""
import argparse
import bisect
import csv
import glob
import os
import re
import sys
import wave

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import cv2
except ImportError:  # optional
    cv2 = None

PHASES = ["Room_Prep", "Patient_Entry", "Sterile_Draping", "Microsurgery", "Emergence"]
PHASE_COLORS = ["#2364AA", "#3DA5D9", "#73BFB8", "#FEC601", "#EA7317"]
ROLES = ["Surgeon", "OTA", "Runner", "Anesthesiologist", "Nurse", "Spectator"]
ROLE_COLORS = ["#C0392B", "#8E44AD", "#16A085", "#2C3E50", "#D35400", "#7F8C8D"]
ROLE_ALIASES = {"ota": "OTA", "assistant": "OTA", "surgicalassistant": "OTA", "surgeon": "Surgeon",
                "runner": "Runner", "anesthesiologist": "Anesthesiologist", "anesth": "Anesthesiologist",
                "anaesthesiologist": "Anesthesiologist", "nurse": "Nurse", "cleaner": "Nurse",
                "spectator": "Spectator", "room": "Room", "ambient": "Room"}
CAMERAS = [f"camera0{i}" for i in range(1, 6)]


# ----------------------------------------------------------------------------- discovery
def find_run_folder(root, run):
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(p) and re.search(rf"Run_0?{run}$", os.path.basename(p)):
            return p
    sys.exit(f"No run folder for Run {run} in {root} (expected rec_*_Run_{run})")


def index_frames(run_folder):
    """{camera: (sorted frame numbers, {frame: path})} from export_full/color."""
    color = os.path.join(run_folder, "export_full", "color")
    frames = {c: {} for c in CAMERAS}
    if os.path.isdir(color):
        with os.scandir(color) as it:
            for e in it:
                m = re.match(r"color_(\d+)_(camera0\d)\.(jpg|jpeg|png)$", e.name, re.I)
                if m and m.group(2) in frames:
                    frames[m.group(2)][int(m.group(1))] = e.path
    return {c: (sorted(d), d) for c, d in frames.items() if d}


def find_videos(run_folder):
    vids = {}
    for c in CAMERAS:
        hits = sorted(glob.glob(os.path.join(run_folder, f"*{c}*.mkv")) + glob.glob(os.path.join(run_folder, "**", f"*{c}*.mkv"), recursive=True))
        if hits:
            vids[c] = hits[0]
    return vids


def find_micro(run_folder):
    """All microscope recordings of a run, in order (Run 1 has Part1 and Part2)."""
    parts = sorted(glob.glob(os.path.join(run_folder, "*Manual*", "video_h265.mp4")))
    if parts:
        return parts
    for ext in ("mp4", "avi", "mov", "mkv"):
        for p in sorted(glob.glob(os.path.join(run_folder, "**", f"*.{ext}"), recursive=True)):
            if re.search(r"basler|micro|mikro", p, re.I):
                return [p]
    return []


def role_from_name(name):
    stem = re.sub(r"(?i)^run_?0?\d+_", "", os.path.splitext(os.path.basename(name))[0])
    for part in re.split(r"[_\-\s]+", stem):
        key = re.sub(r"[^a-z]", "", part.lower())
        if key in ROLE_ALIASES:
            return ROLE_ALIASES[key]
    return None


def find_audio(root, run, audio_dir=None):
    cands = [audio_dir] if audio_dir else (glob.glob(os.path.join(root, "*DVR*", f"Run_{run}")) +
                                          glob.glob(os.path.join(root, "*DVR*", f"Run_{run:02d}")) +
                                          glob.glob(os.path.join(root, "*DVR*", "*", f"Run_{run}")))
    for d in cands:
        if d and os.path.isdir(d):
            files = {}
            for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
                role = role_from_name(p)
                if role and role not in files:
                    files[role] = p
            if files:
                return d, files
    return None, {}


def read_phases(labels, run):
    for p in [os.path.join(labels, "annotations", "phases", f"run{run:02d}.csv"),
              os.path.join(labels, f"run{run:02d}.csv"), os.path.join(labels, f"annotations_{run}.csv"),
              os.path.join(labels, "annotations", f"annotations_{run}.csv")] if labels else []:
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            return [(float(r.get("start_s", r.get("start_sec"))), float(r.get("end_s", r.get("end_sec"))),
                     r.get("phase", r.get("phase_name"))) for r in rows]
    return []


def parse_audacity(path):
    out = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3:
            try:
                out.append((float(parts[0]), float(parts[1]), parts[2].strip()))
            except ValueError:
                pass
    return out


def read_speakers(labels, audio_dir, run, prefer):
    cands = []
    if labels:
        for kind in ([prefer] if prefer != "auto" else ["audited", "pseudo"]):
            cands.append((kind, os.path.join(labels, "annotations", "speakers", kind, f"run{run:02d}.csv")))
    if audio_dir:
        if prefer in ("auto", "audited"):
            cands.append(("audited", os.path.join(audio_dir, f"Run_{run}_Labels_GroundTruth.txt")))
        if prefer in ("auto", "pseudo"):
            cands.append(("pseudo", os.path.join(audio_dir, f"Run_{run}_AntiFragLabels.txt")))
    for kind, p in cands:
        if os.path.exists(p):
            if p.endswith(".csv"):
                rows = list(csv.DictReader(open(p, encoding="utf-8")))
                return kind, [(float(r["start_s"]), float(r["end_s"]), r["speaker"]) for r in rows]
            return kind, parse_audacity(p)
    return None, []


def read_offsets(labels, run):
    """{stream: (offset_s, rate)} for this run from sync/offsets.csv."""
    off = {}
    p = os.path.join(labels, "sync", "offsets.csv") if labels else None
    if p and os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["run"] in (f"run{run:02d}", str(run)) and (r.get("offset_s") or "").strip():
                rate = float(r["rate"]) if (r.get("rate") or "").strip() else 1.0
                off[r["stream"]] = (float(r["offset_s"]), rate)
    return off


# ----------------------------------------------------------------------------- readers
def read_room_times(labels, run, run_folder):
    """{frame: seconds since the first frame}, from sync/room_frame_times/runNN.csv or the
    export's tables_timestamps.arrow (needs pyarrow). None if neither is available."""
    p = os.path.join(labels, "sync", "room_frame_times", f"run{run:02d}.csv") if labels else None
    if p and os.path.exists(p):
        return {int(r["frame"]): float(r["t_s"]) for r in csv.DictReader(open(p, encoding="utf-8"))}
    arrow = os.path.join(run_folder, "export_full", "tables_timestamps.arrow")
    if os.path.exists(arrow):
        try:
            import pyarrow.feather as feather
            tab = feather.read_table(arrow, columns=["frame_number", "camera01_colorimage"])
            fr, ns = tab.column(0).to_pylist(), tab.column(1).to_pylist()
            return {int(f): (n - ns[0]) / 1e9 for f, n in zip(fr, ns)}
        except Exception:  # noqa: BLE001
            pass
    return None


class RoomCameras:
    NOMINAL_FPS = 7.484   # measured average of the exports; used only without frame timestamps

    def __init__(self, run_folder, phases, fps_override=None, times=None):
        self.frames = index_frames(run_folder)
        self.videos = {} if self.frames else find_videos(run_folder)
        self.caps = {}
        self.fps = {}
        self.times = times if (times and not fps_override) else None
        for c, (nums, _) in self.frames.items():
            self.fps[c] = fps_override or self.NOMINAL_FPS
        if self.videos and cv2 is not None:
            for c, p in self.videos.items():
                self.caps[c] = cv2.VideoCapture(p)
        self.cameras = sorted(set(self.frames) | set(self.caps))
        self._tl = {}

    def time_of(self, cam, k):
        if self.times and k in self.times:
            return self.times[k]
        return k / self.fps[cam]

    def frame_time(self, cam, t):
        """Nearest frame number and its time (from the frame timestamps when available)."""
        nums, _ = self.frames[cam]
        if cam not in self._tl:
            self._tl[cam] = [self.time_of(cam, k) for k in nums]
        tl = self._tl[cam]
        i = min(max(bisect.bisect_left(tl, t), 0), len(nums) - 1)
        if i > 0 and abs(tl[i - 1] - t) < abs(tl[i] - t):
            i -= 1
        return nums[i], tl[i]

    def get(self, cam, t, size):
        if cam in self.frames:
            k, _ = self.frame_time(cam, t)
            img = Image.open(self.frames[cam][1][k])
            img.draft("RGB", size)
            img = img.convert("RGB")
            img.thumbnail(size)
            return np.asarray(img)
        if cam in self.caps:
            cap = self.caps[cam]
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
            ok, fr = cap.read()
            if ok:
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                h, w = fr.shape[:2]
                s = min(size[0] / w, size[1] / h)
                return cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return None


class Microscope:
    def __init__(self, path, anchors=None):
        """anchors: (frames, seconds) on the microscope clock, from sync/microscope_frame_times."""
        self.path = path
        self.anchors = anchors
        self.cap = cv2.VideoCapture(path) if (path and cv2 is not None) else None
        self.duration = None
        if self.cap is not None and self.cap.isOpened():
            n = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
            self.duration = n / fps if fps else None
        if anchors:
            self.duration = anchors[1][-1]

    def get(self, t, size):
        if self.cap is None or t < 0 or (self.duration and t > self.duration):
            return None
        if self.anchors:   # real frame times: clock time -> frame number
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(np.interp(t, self.anchors[1], self.anchors[0]))))
        else:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = self.cap.read()
        if not ok:
            return None
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        h, w = fr.shape[:2]
        s = min(size[0] / w, size[1] / h)
        return cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


class MicroscopeSet:
    """The microscope recordings of a run, each placed on the room-camera timeline:
    stream time = (t - offset) * rate."""

    def __init__(self, paths, offsets, rates, anchors=None):
        self.parts = []
        for i, p in enumerate(paths):
            m = Microscope(p, anchors[i] if anchors and i < len(anchors) else None)
            m.name = os.path.basename(os.path.dirname(p)) or f"part {i + 1}"
            m.offset = offsets[i] if i < len(offsets) else None
            m.rate = rates[i] if i < len(rates) else 1.0
            self.parts.append(m)

    @property
    def found(self):
        return any(m.cap is not None for m in self.parts)

    @property
    def placed(self):
        return any(m.cap is not None and m.offset is not None for m in self.parts)

    def get(self, t, size):
        for m in self.parts:
            if m.cap is None or m.offset is None:
                continue
            lt = (t - m.offset) * m.rate
            if 0 <= lt <= (m.duration or 1e12):
                return m.get(lt, size), m
        return None, None

    def describe(self):
        return "; ".join(f"{m.path} (offset {m.offset if m.offset is not None else 'not set'}, "
                         f"{'frame times from the encoder log' if m.anchors else f'rate {m.rate}'})"
                         for m in self.parts) or "not found"


class Wav:
    def __init__(self, path):
        self.path, self.data, self.rate, self.n = path, None, None, 0
        try:
            self.w = wave.open(path, "rb")
            self.rate, self.n = self.w.getframerate(), self.w.getnframes()
            self.ch, self.width = self.w.getnchannels(), self.w.getsampwidth()
        except (wave.Error, EOFError):
            self.w = None
            try:
                from scipy.io import wavfile
                self.rate, self.data = wavfile.read(path, mmap=True)
                self.n = self.data.shape[0]
            except Exception:  # noqa: BLE001
                self.rate = None

    @property
    def duration(self):
        return self.n / self.rate if self.rate else 0

    def window(self, t0, t1):
        """Mono float samples in [-1, 1] between t0 and t1 (seconds of the audio file)."""
        if not self.rate:
            return np.zeros(0), self.rate or 1
        a, b = max(0, int(t0 * self.rate)), min(self.n, int(t1 * self.rate))
        if b <= a:
            return np.zeros(0), self.rate
        if self.w is not None:
            self.w.setpos(a)
            raw = self.w.readframes(b - a)
            if self.width == 2:
                x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            elif self.width == 3:
                u = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
                x = ((u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608
            elif self.width == 4:
                x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
            else:
                x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
            x = x.reshape(-1, self.ch).mean(1)
        else:
            x = np.asarray(self.data[a:b], dtype=np.float32)
            if x.ndim > 1:
                x = x.mean(1)
            if np.issubdtype(self.data.dtype, np.integer):
                x /= float(np.iinfo(self.data.dtype).max)
        return x, self.rate


def envelope(x, bins):
    if len(x) == 0:
        return np.zeros(bins), np.zeros(bins)
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    lo = np.array([x[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([x[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])])
    return lo, hi


# ----------------------------------------------------------------------------- figure
class Composer:
    def __init__(self, a, run_folder, cams, micro, audio_files, phases, spk_kind, speakers, offsets):
        self.a, self.run_folder, self.cams, self.micro = a, run_folder, cams, micro
        self.phases, self.spk_kind, self.speakers = phases, spk_kind, speakers
        self.audio_off = a.audio_offset if a.audio_offset is not None else offsets.get("audio", (0.0, 1.0))
        self.audio_off = self.audio_off[0] if isinstance(self.audio_off, tuple) else self.audio_off
        self.room = Wav(audio_files["Room"]) if "Room" in audio_files else None
        self.run_end = max([e for _, e, _ in phases] + [self.room.duration + self.audio_off if self.room else 0, 1])
        self.fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#111418")
        f = self.fig
        main = a.main
        others = [c for c in CAMERAS if c != main]
        self.ax_main = f.add_axes([0.005, 0.38, 0.555, 0.60])
        self.ax_cams = {main: self.ax_main}
        for i, c in enumerate(others):
            self.ax_cams[c] = f.add_axes([0.565 + (i % 2) * 0.217, 0.785 - (i // 2) * 0.205, 0.212, 0.195])
        self.ax_micro = f.add_axes([0.565, 0.38, 0.212, 0.195])
        self.ax_info = f.add_axes([0.782, 0.38, 0.212, 0.195])
        self.ax_phase = f.add_axes([0.08, 0.305, 0.91, 0.045])
        self.ax_spk = f.add_axes([0.08, 0.105, 0.91, 0.165])
        self.ax_wav = f.add_axes([0.08, 0.025, 0.91, 0.065])
        for ax in [*self.ax_cams.values(), self.ax_micro, self.ax_info]:
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#333a42")
            ax.set_facecolor("#0b0d10")
        self.img = {}
        self._draw_static()

    def _draw_static(self):
        a = self.a
        # phase ribbon over the whole run
        ax = self.ax_phase
        ax.set_facecolor("#1b2027")
        for s, e, p in self.phases:
            i = PHASES.index(p) if p in PHASES else 0
            ax.barh(0, e - s, left=s, height=1, color=PHASE_COLORS[i])
            ax.text((s + e) / 2, 0, p.replace("_", " "), ha="center", va="center", fontsize=9,
                    color="#111" if i in (2, 3) else "white")
        ax.set_xlim(0, self.run_end); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
        ax.tick_params(colors="#9aa4ae", labelsize=8)
        self.ph_line = ax.axvline(0, color="white", lw=2)
        # speaker rows (absolute time; the window is set per frame)
        ax = self.ax_spk
        ax.set_facecolor("#1b2027")
        roles = [r for r in ROLES if any(x[2] == r for x in self.speakers)]
        self.roles = roles or ROLES[:5]
        for i, r in enumerate(self.roles):
            segs = [(s + self.audio_off, e - s) for s, e, rr in self.speakers if rr == r]
            if segs:
                ax.broken_barh(segs, (i - 0.4, 0.8), color=ROLE_COLORS[ROLES.index(r)])
        ax.set_yticks(range(len(self.roles)))
        ax.set_yticklabels(self.roles, color="#d0d6dc", fontsize=9)
        ax.set_ylim(-0.6, len(self.roles) - 0.4)
        ax.invert_yaxis()
        ax.tick_params(colors="#9aa4ae", labelsize=8)
        ax.text(-0.075, 1.02, f"Speakers\n({self.spk_kind or 'none'})", transform=ax.transAxes, color="#9aa4ae",
                fontsize=9, va="bottom")
        self.spk_line = ax.axvline(0, color="white", lw=1.5)
        ax = self.ax_wav
        ax.set_facecolor("#1b2027")
        ax.set_yticks([]); ax.tick_params(colors="#9aa4ae", labelsize=8)
        self.wav_fill = None
        self.wav_line = ax.axvline(0, color="white", lw=1.5)
        ax.set_ylim(-1, 1)
        ax.text(-0.075, 0.5, "Ambient\nmicrophone", transform=ax.transAxes, color="#9aa4ae", fontsize=9, va="center")

    def _show(self, key, ax, arr, label):
        if arr is None:
            arr = np.zeros((9, 16, 3), np.uint8)
        if key not in self.img:
            self.img[key] = ax.imshow(arr, aspect="equal")
            self.lbl = getattr(self, "lbl", {})
            self.lbl[key] = ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", ha="left",
                                    fontsize=10, color="white",
                                    bbox=dict(facecolor="#000000aa", edgecolor="none", pad=2))
        else:
            self.img[key].set_data(arr)
            self.img[key].set_extent((-0.5, arr.shape[1] - 0.5, arr.shape[0] - 0.5, -0.5))
            ax.set_xlim(-0.5, arr.shape[1] - 0.5); ax.set_ylim(arr.shape[0] - 0.5, -0.5)
            self.lbl[key].set_text(label)

    def render(self, t):
        a = self.a
        for c, ax in self.ax_cams.items():
            size = (960, 540) if c == a.main else (480, 270)
            arr = self.cams.get(c, t, size) if c in self.cams.cameras else None
            self._show(c, ax, arr, c.replace("camera0", "Camera "))
        if self.micro.placed:
            arr, part = self.micro.get(t, (480, 362))
            lab = (f"Microscope ({part.name}, offset {part.offset:+.2f} s)" if part is not None
                   else "Microscope: no recording at this time")
        else:
            arr, lab = None, "Microscope: " + ("offset not set" if self.micro.found else "not found")
        self._show("micro", self.ax_micro, arr, lab)
        phase = next((p for s, e, p in self.phases if s <= t < e), "unlabeled")
        self.ax_info.clear(); self.ax_info.set_xticks([]); self.ax_info.set_yticks([])
        self.ax_info.set_facecolor("#0b0d10")
        mm, ss = divmod(t, 60)
        lines = [f"Run {a.run}", f"t = {t:7.2f} s  ({int(mm):02d}:{ss:05.2f})", f"Phase: {phase.replace('_', ' ')}",
                 f"Audio offset: {self.audio_off:+.2f} s", f"Speakers: {self.spk_kind or '-'}"]
        for i, line in enumerate(lines):
            self.ax_info.text(0.05, 0.85 - i * 0.17, line, transform=self.ax_info.transAxes, color="white",
                              fontsize=12 if i < 3 else 10, fontweight="bold" if i == 2 else "normal")
        W = a.window
        self.ph_line.set_xdata([t, t])
        self.ax_spk.set_xlim(t - W, t + W)
        self.spk_line.set_xdata([t, t])
        self.ax_wav.set_xlim(t - W, t + W)
        self.wav_line.set_xdata([t, t])
        if self.wav_fill is not None:
            self.wav_fill.remove()
            self.wav_fill = None
        if self.room is not None:
            x, _ = self.room.window(t - W - self.audio_off, t + W - self.audio_off)
            if len(x):
                peak = max(np.abs(x).max(), 1e-3)
                lo, hi = envelope(x / peak, 1200)
                start = max(t - W, self.audio_off)
                tt = np.linspace(start, start + len(x) / self.room.rate, len(lo))
                self.wav_fill = self.ax_wav.fill_between(tt, lo, hi, color="#5dade2", lw=0)
        return self.fig

    def to_array(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


# ----------------------------------------------------------------------------- sync sheets
def clap_sheet(a, cams, room, audio_off, which, out):
    if room is None:
        sys.exit("No ambient (Room) WAV found for the clap check.")
    span = 180.0
    t0 = 0.0 if which == "start" else max(0.0, room.duration - span)
    x, sr = room.window(t0, t0 + span)
    hop = int(sr * 0.01)
    e = np.array([np.abs(x[i:i + hop]).max() for i in range(0, len(x) - hop, hop)])
    rise = np.diff(e, prepend=e[0])
    k = int(np.argmax(rise))
    t_clap_audio = t0 + k * 0.01
    t_clap = t_clap_audio + audio_off  # on the room-camera timeline
    cam = a.main
    fig = plt.figure(figsize=(19.2, 12), dpi=100)
    axw = fig.add_axes([0.04, 0.82, 0.94, 0.12])
    xs, _ = room.window(t_clap_audio - 2, t_clap_audio + 2)
    tt = np.linspace(-2, 2, len(xs)) if len(xs) else []
    axw.plot(tt, xs, lw=0.6, color="#2c3e50")
    axw.axvline(0, color="#e74c3c")
    axw.set_xlim(-2, 2)
    axw.set_title(f"Run {a.run}: loudest onset in the {'first' if which == 'start' else 'last'} {int(span)} s of the "
                  f"ambient audio at {t_clap_audio:.2f} s (audio) = {t_clap:.2f} s on the room-camera timeline "
                  f"(audio offset {audio_off:+.2f} s). The {cam} frame where the hands meet should be the red one "
                  f"(closest to 0.00 s).", fontsize=10, loc="left")
    if cam in cams.frames:
        nums, paths = cams.frames[cam]
        k0, _ = cams.frame_time(cam, t_clap)
        i0 = nums.index(k0)
        sel = nums[max(0, i0 - 8): i0 + 8]
        times = [cams.time_of(cam, k) for k in sel]
    else:
        fps = 7.5
        times = [t_clap + (i - 8) / fps for i in range(16)]
    best = min(range(len(times)), key=lambda i: abs(times[i] - t_clap))
    for i, tf in enumerate(times):
        ax = fig.add_axes([0.04 + (i % 4) * 0.24, 0.59 - (i // 4) * 0.19, 0.22, 0.165])
        arr = cams.get(cam, tf + 1e-6, (480, 270)) if cam in cams.cameras else None
        if arr is not None:
            ax.imshow(arr)
        ax.set_title(f"{tf - t_clap:+.2f} s  (t = {tf:.2f} s)", fontsize=10,
                     color="#c0392b" if i == best else "black", fontweight="bold" if i == best else "normal")
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out)
    print(f"Clap check written to {out}")


def pair_sheet(a, cams, micro, t, out):
    if not micro.found:
        sys.exit("No microscope video found (use --micro PATH).")
    if not micro.placed:
        sys.exit("No microscope offset yet: pass --micro-offset SECONDS to try a value.")
    times = [t + d for d in np.arange(-2.0, 2.01, 0.5)]
    fig = plt.figure(figsize=(22, 4.8), dpi=100)
    fig.suptitle(f"Run {a.run}: {a.main} (top) and microscope (bottom) at the same timeline times; "
                 f"{micro.describe()}", fontsize=10)
    for i, tt in enumerate(times):
        ax1 = fig.add_axes([0.005 + i * 0.1105, 0.48, 0.108, 0.40])
        ax2 = fig.add_axes([0.005 + i * 0.1105, 0.03, 0.108, 0.43])
        r = cams.get(a.main, tt, (480, 270)) if a.main in cams.cameras else None
        m, _ = micro.get(tt, (480, 362))
        if r is not None:
            ax1.imshow(r)
        if m is not None:
            ax2.imshow(m)
        ax1.set_title(f"t = {tt:.1f} s", fontsize=10)
        for ax in (ax1, ax2):
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out)
    print(f"Pair sheet written to {out}")


def scan_sheet(a, cams, t0, t1, step, out):
    """Main-camera frames every `step` seconds between t0 and t1 (at most 48)."""
    times = list(np.arange(t0, t1 + 1e-9, step))[:48]
    cols = 8
    rows = int(np.ceil(len(times) / cols))
    fig = plt.figure(figsize=(22, 1.6 + 1.75 * rows), dpi=100)
    fig.suptitle(f"Run {a.run}: {a.main}, {t0:.0f}-{t1:.0f} s every {step:g} s (room-camera timeline)", fontsize=12)
    for i, tt in enumerate(times):
        ax = fig.add_axes([0.005 + (i % cols) * 0.1245, 1 - (1.35 + 1.75 * (i // cols + 1)) / (1.6 + 1.75 * rows),
                           0.12, 1.55 / (1.6 + 1.75 * rows)])
        arr = cams.get(a.main, tt, (480, 270)) if a.main in cams.cameras else None
        if arr is not None:
            ax.imshow(arr)
        ax.set_title(f"t = {tt:.1f} s", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(out)
    print(f"Scan sheet written to {out}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="clean dataset folder, e.g. E:\\Ophthalmo_4D")
    ap.add_argument("--run", type=int, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--t", type=float, help="time in seconds on the room-camera timeline")
    mode.add_argument("--clip", type=float, nargs=2, metavar=("T0", "T1"))
    mode.add_argument("--clap", choices=["start", "end"])
    mode.add_argument("--pair", type=float, metavar="T")
    mode.add_argument("--scan", type=float, nargs=2, metavar=("T0", "T1"))
    ap.add_argument("--labels", help="release metadata folder (annotations/, sync/) or a folder with annotations_<n>.csv")
    ap.add_argument("--speakers", choices=["auto", "audited", "pseudo"], default="auto")
    ap.add_argument("--audio-dir", help="folder with the synced WAVs of this run")
    ap.add_argument("--micro", nargs="+", help="microscope video(s) of this run, in order")
    ap.add_argument("--audio-offset", type=float, help="seconds; overrides sync/offsets.csv")
    ap.add_argument("--micro-offset", type=float, nargs="+", help="seconds, one per microscope part; overrides sync/offsets.csv")
    ap.add_argument("--micro-rate", type=float, nargs="+", help="stream seconds per timeline second, one per part (default 1)")
    ap.add_argument("--step", type=float, default=2.0, help="seconds between frames for --scan")
    ap.add_argument("--main", default="camera01", choices=CAMERAS, help="large camera view")
    ap.add_argument("--window", type=float, default=30.0, help="half-width of the speaker/audio window in s")
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second for --clip")
    ap.add_argument("--video-fps", type=float, help="ignore the frame timestamps and use this constant frame rate")
    ap.add_argument("--out", help="output file (default: run<NN>_<mode>_<t>.png/.mp4 in the current folder)")
    a = ap.parse_args()

    run_folder = find_run_folder(a.root, a.run)
    phases = read_phases(a.labels, a.run)
    cams = RoomCameras(run_folder, phases, a.video_fps, read_room_times(a.labels, a.run, run_folder))
    audio_dir, audio_files = find_audio(a.root, a.run, a.audio_dir)
    spk_kind, speakers = read_speakers(a.labels, audio_dir, a.run, a.speakers)
    offsets = read_offsets(a.labels, a.run)
    paths = a.micro or find_micro(run_folder)
    keys = ["microscope"] if len(paths) == 1 else [f"microscope_part{i + 1}" for i in range(len(paths))]
    m_off = a.micro_offset or [offsets[k][0] if k in offsets else None for k in keys]
    m_rate = a.micro_rate or [offsets[k][1] if k in offsets else 1.0 for k in keys]
    anchors = []
    for i in range(len(paths)):
        name = f"run{a.run:02d}.csv" if len(paths) == 1 else f"run{a.run:02d}_part{i + 1}.csv"
        fp = os.path.join(a.labels, "sync", "microscope_frame_times", name) if a.labels else ""
        if fp and os.path.exists(fp) and not a.micro_rate:
            rows = list(csv.DictReader(open(fp, encoding="utf-8")))
            anchors.append(([int(r["frame"]) for r in rows], [float(r["t_s"]) for r in rows]))
        else:
            anchors.append(None)
    micro = MicroscopeSet(paths, m_off, m_rate, anchors)
    print(f"Run folder: {run_folder}")
    print(f"Room cameras: {', '.join(f'{c} ({len(cams.frames[c][0])} frames)' for c in cams.frames) or ', '.join(cams.caps) or 'none found'}; "
          f"frame times: {'timestamps' if cams.times else f'constant {RoomCameras.NOMINAL_FPS} fps (no timestamps found)'}")
    print(f"Microscope: {micro.describe()} | Audio: {audio_dir or 'not found'} {sorted(audio_files)}")
    print(f"Phases: {len(phases)} segments | Speakers: {spk_kind or 'none'} ({len(speakers)} segments) | Offsets: {offsets or 'none'}")

    audio_off = a.audio_offset if a.audio_offset is not None else offsets.get("audio", (0.0, 1.0))[0]
    if a.clap:
        out = a.out or f"run{a.run:02d}_clap_{a.clap}.png"
        clap_sheet(a, cams, Wav(audio_files["Room"]) if "Room" in audio_files else None, audio_off, a.clap, out)
        return
    if a.pair is not None:
        pair_sheet(a, cams, micro, a.pair, a.out or f"run{a.run:02d}_pair_{a.pair:.0f}.png")
        return
    if a.scan:
        scan_sheet(a, cams, a.scan[0], a.scan[1], a.step, a.out or f"run{a.run:02d}_scan_{a.scan[0]:.0f}-{a.scan[1]:.0f}.png")
        return
    comp = Composer(a, run_folder, cams, micro, audio_files, phases, spk_kind, speakers, offsets)
    if a.t is not None:
        out = a.out or f"run{a.run:02d}_t{a.t:07.1f}.png"
        comp.render(a.t).savefig(out, facecolor=comp.fig.get_facecolor())
        print(f"Written: {out}")
        return
    t0, t1 = a.clip
    times = np.arange(t0, t1, 1.0 / a.fps)
    out = a.out or f"run{a.run:02d}_{t0:.0f}-{t1:.0f}.mp4"
    writer = None
    if cv2 is not None:
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (1920, 1080))
    else:
        out = os.path.splitext(out)[0]
        os.makedirs(out, exist_ok=True)
        print("opencv-python not installed: writing PNG frames instead of an MP4.")
    for i, t in enumerate(times):
        comp.render(float(t))
        arr = comp.to_array()
        if writer is not None:
            writer.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(arr).save(os.path.join(out, f"frame_{i:05d}.png"))
        if i % 10 == 0:
            print(f"  {i + 1}/{len(times)}  t = {t:.1f} s", flush=True)
    if writer is not None:
        writer.release()
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
