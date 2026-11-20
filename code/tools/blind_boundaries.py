"""Bifocal-OR: blinded second annotation of the phase boundaries, and its agreement score.

Every run has four internal boundaries (Room_Prep|Patient_Entry, Patient_Entry|Sterile_Draping,
Sterile_Draping|Microsurgery, Microsurgery|Emergence), 36 in total. For each one, `prepare` cuts a
window of --window seconds (default 180) out of the five room cameras at one frame per second,
plus the ambient audio. The window is shifted by a random offset (up to 30 % of its length), so
the reference boundary lies anywhere between 20 % and 80 % of the window and its position cannot
be guessed. The 36 windows are shuffled and shown under neutral IDs (T01-T36) without run numbers
or absolute times. You open annotate.html in Chrome or Edge, find the first second of the new
phase in each window, and download answers.csv. `score` then compares your boundaries with the
reference labels.

1) Prepare (about 10-20 min; writes about 1.5-2 GB, so NOT onto the data drive):
    python blind_boundaries.py prepare --root "E:\\Ophthalmo_4D" --sync-audio "E:\\Ophthalmo_4D\\Ophthalmo_4D_DVR_Sync"
        --labels "<release folder Bifocal-OR>" --rules rules.txt --out "<Desktop>\\blind_boundaries"
   --root        folder with the rec_*_Run_<n> folders (frames in export_full/color/)
   --sync-audio  folder with Run_<n>/ subfolders holding the synced WAVs (the Room/ambient track is used)
   --labels      release metadata folder: annotations/phases/runNN.csv, sync/room_frame_times/runNN.csv,
                 sync/offsets.csv
   --rules       text file with the boundary rules shown next to every task (Mateus's rules);
                 without it, the short definitions of the release README are shown
   Re-run with --html-only to rebuild annotate.html after editing the rules (keeps the clips).
   Do not open _key.b64 in the output folder: it holds the answers.

2) Annotate: open <out>/annotate.html. Keys: Left/Right = -1/+1 s, Shift+Left/Right = -10/+10 s,
   Space = play/pause (with audio), B = boundary here, U = unsure, N = next task. Progress is kept
   in the browser; "Download answers.csv" saves the result (also do this before closing).

3) Score:
    python blind_boundaries.py score --out "<Desktop>\\blind_boundaries" --answers answers.csv
   Writes agreement_per_boundary.csv and agreement_summary.md (absolute deviations, share within
   5/10/30 s, frame-level accuracy and Cohen's kappa over all labeled seconds of the nine runs).

Needs: numpy, pillow.  pip install numpy pillow
"""
import argparse
import base64
import bisect
import csv
import glob
import json
import os
import random
import re
import sys
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

PHASES = ["Room_Prep", "Patient_Entry", "Sterile_Draping", "Microsurgery", "Emergence"]
CAMERAS = [f"camera0{i}" for i in range(1, 6)]
DEFAULT_RULES = """Boundary = the first second that belongs to the NEW phase.
Room_Prep -> Patient_Entry: patient transfer into the OR begins.
Patient_Entry -> Sterile_Draping: sterile preparation and draping begins (see the playbook).
Sterile_Draping -> Microsurgery: right after the team time-out.
Microsurgery -> Emergence: undraping begins."""


# ----------------------------------------------------------------------------- inputs
def read_phases(labels, run):
    for p in (os.path.join(labels, "annotations", "phases", f"run{run:02d}.csv"),
              os.path.join(labels, f"annotations_{run}.csv")):
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            return [(r.get("phase", r.get("phase_name")), float(r.get("start_s", r.get("start_sec"))),
                     float(r.get("end_s", r.get("end_sec")))) for r in rows]
    return None


def read_frame_times(labels, run):
    p = os.path.join(labels, "sync", "room_frame_times", f"run{run:02d}.csv")
    if not os.path.exists(p):
        return None
    return {int(r["frame"]): float(r["t_s"]) for r in csv.DictReader(open(p, encoding="utf-8"))}


def read_audio_offset(labels, run):
    p = os.path.join(labels, "sync", "offsets.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["run"] == f"run{run:02d}" and r["stream"] == "audio" and r["offset_s"].strip():
                return float(r["offset_s"])
    return None


def find_run_folder(root, run):
    for p in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(p) and re.search(rf"Run_0?{run}$", os.path.basename(p)):
            return p
    return None


def index_frames(run_folder):
    color = os.path.join(run_folder, "export_full", "color")
    frames = {c: {} for c in CAMERAS}
    if os.path.isdir(color):
        with os.scandir(color) as it:
            for e in it:
                m = re.match(r"color_(\d+)_(camera0\d)\.(jpg|jpeg|png)$", e.name, re.I)
                if m and m.group(2) in frames:
                    frames[m.group(2)][int(m.group(1))] = e.path
    return {c: d for c, d in frames.items() if d}


def find_room_wav(sync_audio, run):
    for d in (os.path.join(sync_audio, f"Run_{run}"), os.path.join(sync_audio, f"Run_{run:02d}")):
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            if re.search(r"room|ambient", os.path.basename(p), re.I):
                return p
    return None


def wav_segment(path, t0, t1, out, sr_out=16000):
    """Mono 16-bit WAV of [t0, t1] s of the file at `path` (silence outside the file)."""
    with wave.open(path, "rb") as w:
        sr, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        a, b = int(round(t0 * sr)), int(round(t1 * sr))
        lo, hi = max(a, 0), min(b, n)
        x = np.zeros(b - a, np.float32)
        if hi > lo:
            w.setpos(lo)
            raw = w.readframes(hi - lo)
            if width == 2:
                y = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            elif width == 4:
                y = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
            elif width == 3:
                u = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
                y = ((u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608
            else:
                y = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
            y = y.reshape(-1, ch).mean(1)
            x[lo - a: lo - a + len(y)] = y
    step = max(1, sr // sr_out)
    if step > 1:   # box low-pass, then decimate (good enough for listening)
        x = np.convolve(x, np.ones(step) / step, mode="same")[::step]
    peak = float(np.abs(x).max()) if len(x) else 0
    if peak > 0:
        x = x / peak * 0.9
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr // step)
        w.writeframes((x * 32767).astype("<i2").tobytes())


def thumb(src, dst, width):
    img = Image.open(src)
    img.draft("RGB", (width, width))
    img = img.convert("RGB")
    h = round(img.height * width / img.width)
    img.resize((width, h), Image.BILINEAR).save(dst, quality=80)


# ----------------------------------------------------------------------------- prepare
def prepare(a):
    rules = open(a.rules, encoding="utf-8").read() if a.rules else DEFAULT_RULES
    os.makedirs(a.out, exist_ok=True)
    key_path = os.path.join(a.out, "_key.b64")
    if a.html_only:
        if not os.path.exists(key_path):
            sys.exit("--html-only needs an existing output folder (run prepare first)")
        key = json.loads(base64.b64decode(open(key_path).read()))
        write_html(a.out, key, rules)
        print(f"Rebuilt {os.path.join(a.out, 'annotate.html')}")
        return
    if os.path.exists(key_path) and not a.force:
        sys.exit(f"{a.out} already holds a prepared set. Use another --out, or --force to redo it "
                 f"(redoing draws new offsets; do it only before you start annotating).")

    rng = random.Random(a.seed)
    tasks = []
    for run in a.runs:
        ph = read_phases(a.labels, run)
        if not ph:
            print(f"Run {run}: no phase labels found, skipped")
            continue
        times = read_frame_times(a.labels, run)
        folder = find_run_folder(a.root, run)
        if folder is None:
            print(f"Run {run}: no rec_*_Run_{run} folder in {a.root}, skipped")
            continue
        frames = index_frames(folder)
        cams = [c for c in a.cameras if c in frames]
        if a.main in cams:
            cams = [a.main] + [c for c in cams if c != a.main]
        if not cams:
            print(f"Run {run}: no color frames in {folder}\\export_full\\color, skipped")
            continue
        tl = {}
        for c in cams:
            nums = sorted(frames[c])
            tl[c] = (nums, [times[k] if times and k in times else k / 7.484 for k in nums])
        run_end = max(v[1][-1] for v in tl.values())
        wav = find_room_wav(a.sync_audio, run) if a.sync_audio else None
        off = read_audio_offset(a.labels, run)
        if a.sync_audio and (wav is None or off is None):
            print(f"Run {run}: no ambient WAV or audio offset found, tasks of this run have no sound")
        for k in range(1, len(ph)):
            b = ph[k][1]
            u = rng.uniform(-0.3, 0.3) * a.window
            start = int(round(b - a.window / 2 + u))
            start = min(max(start, 0), int(run_end) - a.window)
            tasks.append(dict(run=run, k=k, before=ph[k - 1][0], after=ph[k][0], ref=b, start=start, phases=ph,
                              n=a.window, cams=cams, frames=tl, folder=frames,
                              wav=wav if (wav and off is not None) else None, off=off))
        print(f"Run {run}: {len(ph) - 1} boundaries, cameras {', '.join(cams)}, audio {'yes' if wav and off is not None else 'no'}")
    if not tasks:
        sys.exit("Nothing to prepare.")
    rng.shuffle(tasks)
    jobs = []
    for i, t in enumerate(tasks):
        t["id"] = f"T{i + 1:02d}"
        d = os.path.join(a.out, "clips", t["id"])
        os.makedirs(d, exist_ok=True)
        for c in t["cams"]:
            nums, tt = t["frames"][c]
            for s in range(t["n"]):
                target = t["start"] + s + 0.5   # middle of the labeled second
                j = bisect.bisect_left(tt, target)
                j = min(max(j, 0), len(nums) - 1)
                if j > 0 and abs(tt[j - 1] - target) < abs(tt[j] - target):
                    j -= 1
                jobs.append((t["folder"][c][nums[j]], os.path.join(d, f"{c}_{s:03d}.jpg"),
                             a.main_width if c == a.main else a.width))
        if t["wav"]:
            wav_segment(t["wav"], t["start"] - t["off"], t["start"] + t["n"] - t["off"], os.path.join(d, "audio.wav"))
    print(f"{len(tasks)} tasks, {len(jobs)} frames to write ...")
    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for _ in ex.map(lambda j: thumb(*j), jobs):
            done += 1
            if done % 2000 == 0:
                print(f"  {done}/{len(jobs)}")
    key = [dict(id=t["id"], run=t["run"], k=t["k"], before=t["before"], after=t["after"], ref=t["ref"],
                start=t["start"], n=t["n"], cams=t["cams"], audio=bool(t["wav"]), phases=t["phases"]) for t in tasks]
    open(key_path, "w").write(base64.b64encode(json.dumps(key).encode()).decode())
    write_html(a.out, key, rules)
    print(f"Done. Open {os.path.join(a.out, 'annotate.html')} in Chrome or Edge.")


def write_html(out, key, rules):
    pub = [dict(id=t["id"], before=t["before"], after=t["after"], n=t["n"], cams=t["cams"], audio=t["audio"])
           for t in key]
    html = HTML.replace("__TASKS__", json.dumps(pub)).replace("__RULES__", json.dumps(rules))
    open(os.path.join(out, "annotate.html"), "w", encoding="utf-8").write(html)


# ----------------------------------------------------------------------------- score
def kappa(a, b, k=5):
    a, b = np.asarray(a), np.asarray(b)
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in range(k))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(a):
    key = {t["id"]: t for t in json.loads(base64.b64decode(open(os.path.join(a.out, "_key.b64")).read()))}
    ans = {}
    for r in csv.DictReader(open(a.answers, encoding="utf-8-sig")):
        if (r.get("boundary_s") or "").strip():
            ans[r["task"]] = (int(float(r["boundary_s"])), r.get("unsure", "") in ("1", "yes", "true"), r.get("comment", ""))
    rows = []
    for tid, t in sorted(key.items(), key=lambda kv: (kv[1]["run"], kv[1]["k"])):
        if tid not in ans:
            continue
        s, unsure, comment = ans[tid]
        second = t["start"] + s
        rows.append(dict(task=tid, run=t["run"], transition=f"{t['before']}|{t['after']}", reference_s=t["ref"],
                         second_annotator_s=second, diff_s=round(second - t["ref"], 1), unsure=int(unsure), comment=comment))
    if not rows:
        sys.exit("No answers found.")
    with open(os.path.join(a.out, "agreement_per_boundary.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def stats(sel):
        d = np.abs([r["diff_s"] for r in sel])
        return (len(d), float(np.mean(d)), float(np.median(d)), float(d.max()),
                100 * float((d <= 5).mean()), 100 * float((d <= 10).mean()), 100 * float((d <= 30).mean()))

    lines = ["# Blinded second annotation of the phase boundaries", "",
             f"{len(rows)} of {len(key)} boundaries answered ({sum(r['unsure'] for r in rows)} marked unsure).", "",
             "| Transition | n | mean abs. dev. (s) | median (s) | max (s) | <=5 s | <=10 s | <=30 s | mean signed (s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups = [(f"{PHASES[k - 1]} -> {PHASES[k]}", [r for r in rows if r["transition"] == f"{PHASES[k - 1]}|{PHASES[k]}"])
              for k in range(1, 5)] + [("All", rows)]
    for name, sel in groups:
        if sel:
            n, mean, med, mx, p5, p10, p30 = stats(sel)
            signed = float(np.mean([r["diff_s"] for r in sel]))
            lines.append(f"| {name} | {n} | {mean:.1f} | {med:.1f} | {mx:.1f} | {p5:.0f}% | {p10:.0f}% | {p30:.0f}% | {signed:+.1f} |")
    # frame-level agreement over the labeled seconds of runs where all four boundaries were answered
    ref_all, sec_all, per_run = [], [], []
    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], {})[key[r["task"]]["k"]] = r["second_annotator_s"]
    phases_of = {t["run"]: t["phases"] for t in key.values()}
    for run, bd in sorted(by_run.items()):
        if len(bd) < 4:
            continue
        ph = phases_of[run]
        secs = np.arange(np.ceil(ph[0][1]), np.floor(ph[-1][2])) + 0.5
        ref = np.searchsorted(np.array([s for _, s, _ in ph[1:]]), secs, side="right")
        cuts = sorted(bd[k] for k in range(1, 5))
        sec = np.searchsorted(np.array(cuts, float), secs, side="right")
        po, kp = kappa(ref, sec)
        per_run.append((run, po, kp))
        ref_all.append(ref)
        sec_all.append(sec)
    if ref_all:
        po, kp = kappa(np.concatenate(ref_all), np.concatenate(sec_all))
        lines += ["", f"Frame-level agreement (1-s samples, {len(per_run)} runs with all four boundaries): "
                      f"accuracy {po:.3f}, Cohen's kappa {kp:.3f}.", "",
                  "| Run | accuracy | kappa |", "| --- | --- | --- |"]
        lines += [f"| {r} | {p:.3f} | {k:.3f} |" for r, p, k in per_run]
        n, mean, med, mx, p5, p10, p30 = stats(rows)
        lines += ["", "Suggested sentence for limitation (vii):", "",
                  f"> A second annotator re-annotated all {n} internal phase boundaries blinded to the reference "
                  f"labels, each within a {key[rows[0]['task']]['n'] // 60}-min window placed at a random offset around the "
                  f"boundary; the median absolute deviation was {med:.0f}~s ({p10:.0f}\\% within 10~s), and the "
                  f"frame-level agreement was {100 * po:.1f}\\% (Cohen's $\\kappa = {kp:.2f}$)."]
    open(os.path.join(a.out, "agreement_summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------- page
HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Blind boundary annotation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1d232a;--muted:#5f6b76;--line:#d9dee3;--accent:#2364AA;--ok:#2e8b57;--warn:#d98e04}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
#app{display:grid;grid-template-columns:190px 1fr;height:100vh}
#side{border-right:1px solid var(--line);background:var(--panel);overflow:auto;padding:8px}
#side h2{font-size:13px;margin:4px 4px 8px;color:var(--muted)}
.t{display:flex;justify-content:space-between;padding:5px 8px;border-radius:6px;cursor:pointer}
.t:hover{background:#eef2f6}.t.cur{background:#dfe9f5;font-weight:600}.t .s{font-size:12px;color:var(--muted)}
.t .s.done{color:var(--ok)}.t .s.unsure{color:var(--warn)}
#main{display:flex;flex-direction:column;overflow:hidden}
#head{padding:10px 14px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
#head .tr{font-size:18px;font-weight:600}#head .rule{color:var(--muted);white-space:pre-wrap;max-width:900px}
#grid{flex:1;display:grid;grid-template-columns:2fr 1fr 1fr;gap:6px;padding:8px;overflow:auto;align-content:start}
#grid figure.main{grid-row:span 2}#grid.one figure.big{grid-row:auto}
#grid figure{margin:0;background:#000;border-radius:4px;overflow:hidden;position:relative;cursor:zoom-in}
#grid img{width:100%;display:block}#grid figcaption{position:absolute;left:6px;top:4px;color:#fff;font-size:12px;text-shadow:0 0 3px #000}
#grid.one{grid-template-columns:1fr}#grid.one figure{cursor:zoom-out}#grid.one figure:not(.big){display:none}
#ctrl{background:var(--panel);border-top:1px solid var(--line);padding:8px 14px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
#slider{flex:1 1 400px}button{font:inherit;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}#time{font-variant-numeric:tabular-nums;min-width:110px}
#ans{font-weight:600}#marker{position:relative;height:6px;margin:0 14px;background:#e9edf1}#marker i{position:absolute;top:0;width:3px;height:6px;background:var(--accent)}
details{margin:6px 4px;color:var(--muted);font-size:12px}
</style></head><body><div id="app"><div id="side"><h2>Tasks</h2><div id="list"></div>
<details><summary>Rules</summary><div id="rulesAll" style="white-space:pre-wrap"></div></details>
<div style="padding:4px"><button class="primary" id="dl" style="width:100%">Download answers.csv</button>
<label style="display:block;margin-top:8px;font-size:12px;color:var(--muted)">Restore from answers.csv<input type="file" id="imp" accept=".csv" style="width:100%"></label></div></div>
<div id="main"><div id="head"><span class="tr" id="tr"></span><span class="rule" id="rule"></span></div>
<div id="grid"></div><div id="marker"><i id="mk" style="display:none"></i></div>
<div id="ctrl"><button id="m10">&laquo; 10 s</button><button id="m1">&lsaquo; 1 s</button>
<input type="range" id="slider" min="0" value="0"><button id="p1">1 s &rsaquo;</button><button id="p10">10 s &raquo;</button>
<span id="time"></span><button id="play">Play</button><select id="rate"><option>1</option><option selected>2</option><option>4</option><option>8</option></select>&times;
<button class="primary" id="mark">Boundary here (B)</button><span id="ans">no answer</span>
<label><input type="checkbox" id="unsure"> unsure (U)</label><input id="comment" placeholder="comment (Enter to finish)" style="width:180px">
<button id="next">Next task (N)</button></div></div></div><audio id="audio" preload="auto"></audio>
<script>
const TASKS=__TASKS__, RULES=__RULES__, KEY="bifocal_blind_v1";
let A={}; try{A=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(A))}catch(e){}}
let cur=0,sec=0,playing=false,timer=null;
const $=id=>document.getElementById(id), audio=$("audio");
$("rulesAll").textContent=RULES;
function nice(p){return p.replace("_"," ")}
function norm(s){return s.toLowerCase().replace(/[_\s-]+/g," ")}
function ruleFor(t){const ls=RULES.split("\n"),b=norm(t.before),a=norm(t.after);
  const hit=ls.filter(l=>norm(l).includes(b)&&norm(l).includes(a));
  const gen=ls.filter(l=>l.trim()&&!["room prep","patient entry","sterile draping","microsurgery","emergence"].some(p=>norm(l).includes(p)));
  return hit.length?gen.concat(hit).join("\n"):RULES}
function list(){const L=$("list");L.innerHTML="";TASKS.forEach((t,i)=>{const a=A[t.id]||{};const d=document.createElement("div");
  d.className="t"+(i===cur?" cur":"");const st=a.s==null?"–":(a.u?"? "+a.s+" s":a.s+" s");
  d.innerHTML=`<span>${t.id}</span><span class="s ${a.s==null?"":(a.u?"unsure":"done")}">${st}</span>`;d.onclick=()=>open(i);L.appendChild(d)})}
function open(i){stop();cur=i;const t=TASKS[i];$("tr").textContent=t.id+" · "+nice(t.before)+" → "+nice(t.after);
  $("rule").textContent=ruleFor(t);const g=$("grid");g.innerHTML="";g.classList.remove("one");
  t.cams.forEach((c,ci)=>{const f=document.createElement("figure");if(ci===0&&t.cams.length>1)f.className="main";f.innerHTML=`<img data-c="${c}"><figcaption>${c.replace("camera0","Camera ")}</figcaption>`;
   f.onclick=()=>{if(g.classList.contains("one")){g.classList.remove("one");f.classList.remove("big")}else{f.classList.add("big");g.classList.add("one")}};g.appendChild(f)});
  $("slider").max=t.n-1;audio.src=t.audio?`clips/${t.id}/audio.wav`:"";const a=A[t.id]||{};
  $("unsure").checked=!!a.u;$("comment").value=a.c||"";show(a.s!=null?a.s:0);list();
  for(let s=0;s<t.n;s+=5)t.cams.forEach(c=>{new Image().src=`clips/${t.id}/${c}_${String(s).padStart(3,"0")}.jpg`})}
function show(s){const t=TASKS[cur];sec=Math.max(0,Math.min(t.n-1,s|0));$("slider").value=sec;
  document.querySelectorAll("#grid img").forEach(im=>{im.src=`clips/${t.id}/${im.dataset.c}_${String(sec).padStart(3,"0")}.jpg`});
  $("time").textContent=`t = ${sec} s / ${t.n}`;const a=A[t.id]||{};
  $("ans").textContent=a.s==null?"no answer":`boundary at ${a.s} s`;const mk=$("mk");
  if(a.s==null)mk.style.display="none";else{mk.style.display="block";mk.style.left=(100*a.s/(t.n-1))+"%"}}
function mark(){const t=TASKS[cur];A[t.id]=Object.assign(A[t.id]||{},{s:sec});save();show(sec);list()}
function stop(){playing=false;clearInterval(timer);audio.pause();$("play").textContent="Play"}
function play(){if(playing){stop();return}const t=TASKS[cur];playing=true;$("play").textContent="Pause";const r=+$("rate").value;
  if(t.audio){audio.playbackRate=r;audio.currentTime=sec;audio.play().catch(()=>{});
   timer=setInterval(()=>{show(Math.floor(audio.currentTime));if(audio.ended)stop()},100)}
  else timer=setInterval(()=>{if(sec>=t.n-1)stop();else show(sec+1)},1000/r)}
$("slider").oninput=e=>{show(+e.target.value);if(playing&&TASKS[cur].audio)audio.currentTime=sec};
$("m10").onclick=()=>show(sec-10);$("m1").onclick=()=>show(sec-1);$("p1").onclick=()=>show(sec+1);$("p10").onclick=()=>show(sec+10);
$("play").onclick=play;$("mark").onclick=mark;$("next").onclick=()=>open((cur+1)%TASKS.length);
$("unsure").onchange=e=>{const t=TASKS[cur];A[t.id]=Object.assign(A[t.id]||{},{u:e.target.checked});save();list()};
$("comment").oninput=e=>{const t=TASKS[cur];A[t.id]=Object.assign(A[t.id]||{},{c:e.target.value});save()};
$("comment").onkeydown=e=>{if(e.key==="Enter"||e.key==="Escape")e.target.blur()};
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT"&&e.target.type!=="range"&&e.target.type!=="checkbox")return;
  if(e.key==="ArrowLeft"){show(sec-(e.shiftKey?10:1));e.preventDefault()}else if(e.key==="ArrowRight"){show(sec+(e.shiftKey?10:1));e.preventDefault()}
  else if(e.key===" "){play();e.preventDefault()}else if(e.key==="b"||e.key==="B")mark();
  else if(e.key==="u"||e.key==="U"){$("unsure").checked=!$("unsure").checked;$("unsure").onchange({target:$("unsure")})}
  else if(e.key==="n"||e.key==="N")open((cur+1)%TASKS.length)});
function csvq(s){s=String(s==null?"":s);return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
$("dl").onclick=()=>{const rows=["task,boundary_s,unsure,comment"].concat(TASKS.map(t=>{const a=A[t.id]||{};
  return [t.id,a.s==null?"":a.s,a.u?1:0,csvq(a.c)].join(",")}));const b=new Blob([rows.join("\n")+"\n"],{type:"text/csv"});
  const l=document.createElement("a");l.href=URL.createObjectURL(b);l.download="answers.csv";document.body.appendChild(l);l.click();l.remove()};
$("imp").onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(tx=>{tx.split(/\r?\n/).slice(1).forEach(line=>{
  const m=line.match(/^([^,]+),([^,]*),([^,]*),?(.*)$/);if(!m)return;let c=m[4]||"";if(c.startsWith('"'))c=c.slice(1,-1).replace(/""/g,'"');
  A[m[1]]={s:m[2]===""?null:+m[2],u:m[3]==="1",c:c}});save();open(cur)})};
open(0);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sync-audio", help="folder with Run_<n>/ subfolders of synced WAVs (optional)")
    p.add_argument("--rules", help="text file with the boundary rules (optional)")
    p.add_argument("--runs", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--cameras", nargs="+", default=CAMERAS)
    p.add_argument("--window", type=int, default=180, help="seconds per task (default 180)")
    p.add_argument("--width", type=int, default=480, help="thumbnail width of the other cameras in pixels")
    p.add_argument("--main", default="camera01", help="camera shown large (the boundaries were annotated on camera01)")
    p.add_argument("--main-width", type=int, default=1280, help="thumbnail width of the main camera in pixels")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--html-only", action="store_true", help="only rebuild annotate.html (e.g. after editing the rules)")
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True, help="the folder written by prepare")
    s.add_argument("--answers", required=True)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else score(a)


if __name__ == "__main__":
    main()
