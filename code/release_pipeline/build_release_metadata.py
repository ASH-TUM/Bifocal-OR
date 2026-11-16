"""Build the Bifocal-OR metadata layer: annotations, calibration, run metadata.

Reads the raw annotation folder (Bifocal_OR_annotations_raw) and writes the
small, text-only part of the release: everything except the recordings.
Every number the paper reports about these files is re-checked on the way;
the results go to VALIDATION.md.

Usage:
    python build_release_metadata.py RAW_DIR OUT_DIR [SYNC_DIR]

SYNC_DIR (optional) is the extracted sync_files.zip from collect_sync_files.py; it adds the
far-field pyannote outputs of Runs 3, 4, 7 (checked against Table V) and the per-run
pseudo-label diagnostics.

RAW_DIR must contain (as collected on 9 Nov 2026):
    mateus_code/bifocal-or-tecno/annotations/annotations_<n>.csv   phase labels (paper version)
    speakers_audited/Run_<n>_Labels_GroundTruth.txt                 Runs 3, 4, 7
    speakers_pseudo/Run_<n>_AntiFragLabels.txt / .csv               Runs 1-9
    calibration/camera0<c>.json, *_extrinsics.json
    docs/ArUco_Mappings.xlsx, Workflow_Clean.xlsx, aruco_config.json,
         Microphones_Missing_Mappings.xlsx, Director_annotations.xlsx

Needs numpy, scipy and openpyxl.
"""
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import openpyxl
from scipy.optimize import linear_sum_assignment

RUNS = range(1, 10)
PHASES = ["Room_Prep", "Patient_Entry", "Sterile_Draping", "Microsurgery", "Emergence"]
ROLES = ["Surgeon", "OTA", "Runner", "Anesthesiologist", "Nurse", "Spectator"]
RUN_FOLDERS = {  # from build_dataframe.py; folder times = recording PCs' clock (director's log is 1 h later)
    1: "rec_12-03-2026_11-25_Run_1", 2: "rec_12-03-2026_12-40_Run_2",
    3: "rec_12-03-2026_14-02_Run_3", 4: "rec_12-03-2026_14-52_Run_4",
    5: "rec_13-03-2026_09-32_Run_5", 6: "rec_13-03-2026_10-42_Run_6",
    7: "rec_13-03-2026_13-26_Run_7", 8: "rec_13-03-2026_14-19_Run_8",
    9: "rec_13-03-2026_15-37_Run_9",
}
AUDITED = (3, 4, 7)
# Paper, Table II: phase seconds on the evaluated camera-3 frames (RP, PE, SD, MS, EM)
PAPER_TABLE_II = {1: [206, 510, 507, 509, 227], 2: [208, 391, 404, 666, 233], 3: [220, 350, 377, 581, 247],
                  4: [215, 332, 393, 550, 210], 5: [221, 346, 682, 729, 295], 6: [234, 295, 621, 768, 272],
                  7: [197, 271, 443, 689, 214], 8: [192, 307, 479, 1096, 249], 9: [184, 288, 545, 1059, 388]}
# Paper, Table V, wearable reference (FA, Miss, Conf, DER), collar 0.25 s, Run 3 without Anesthesiologist
PAPER_TABLE_V = {3: (8.5, 20.6, 10.4, 39.5), 4: (12.1, 28.3, 8.9, 49.2), 7: (11.4, 28.4, 17.7, 57.6)}
# Paper, Table V, ambient room microphone (pyannote.audio, untuned)
PAPER_TABLE_V_AMBIENT = {3: (4.3, 40.4, 10.0, 54.7), 4: (3.6, 59.6, 13.7, 76.9), 7: (3.0, 65.1, 9.2, 77.2)}  # Run 4 corrected 10 Nov (T14)

# Curated, role-based notes from the director's log (no names). Reviewed by the authors before release.
RUN_NOTES = {
    1: ["Wearable microphones of Surgeon and Anesthesiologist did not record",
        "Phaco machine alarm (beeping); microscope could not be moved as intended",
        "Several breaches of the simulated sterile field (items dropped on the floor)"],
    2: ["The supervising ophthalmic surgeon was in the OR (wearing markers) and gave directions"],
    3: ["Lights briefly switched on during Microsurgery while a visitor took photos (~15:30:53 to 15:31:19)",
        "Attending surgeon entered briefly to check the procedure",
        "Cleaner changes to the Nurse role; first gowning attempt failed (see workflow playbook)"],
    4: ["Visitors: the attending surgeon joined (~16:11), a senior researcher joined and talked with the team (~16:19), "
        "another visitor entered briefly (~16:22)",
        "Cleaner lost the back marker (~15:54); another team member briefly lost the front marker"],
    5: ["Runner lost the front marker (~10:36 and ~10:54)",
        "Microscope problems; the microscope had to be repositioned after surgery",
        "Door closed because of noise from outside"],
    6: ["Eye model forgotten and placed on the table at ~11:58; iodine preparation of the eye repeated"],
    7: ["Noise from people outside the OR"],
    8: ["A trainee observed the team as spectator and wore an additional wearable microphone"],
    9: ["Ex-vivo porcine eye",
        "Surgeon's wearable microphone did not record",
        "A new participant took the cleaner/assistant role",
        "The director moved around the OR taking photos and videos",
        "Anesthesiologist briefly lost a marker; a lens was dropped and searched for with phone flashlights"],
}
MILESTONES = [("Start Time", "first_person_in"), ("Patient comes in", "patient_in"), ("Time Out", "time_out"),
              ("Lights out", "lights_out"), ("Lights on", "lights_on"), ("Patient moved out", "patient_out"),
              ("End Time", "last_person_out")]

checks = []  # (status, text)


def check(ok, text):
    checks.append(("PASS" if ok else "CHECK", text))
    print(("  ok   " if ok else "  !!   ") + text)


def write_csv(path, header, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)


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


def der_collar(ref, hyp, collar=0.25, ignore=(), frame=0.01):
    """DER with pyannote's collar convention (half before, half after each reference
    boundary), optimal one-to-one mapping, overlap-aware, 10 ms grid."""
    tmax = max(e for _, e, _ in ref + hyp)
    n = int(math.ceil(tmax / frame)) + 2
    uem = np.ones(n, bool)
    for s, e, r in ref:
        if r in ignore:
            uem[int(round(s / frame)):int(round(e / frame))] = False
    ref = [x for x in ref if x[2] not in ignore]
    hyp = [x for x in hyp if x[2] not in ignore]
    for s, e, _ in ref:
        for b in (s, e):
            uem[max(0, int(round((b - collar / 2) / frame))):int(round((b + collar / 2) / frame))] = False

    def grid(labels):
        roles = sorted({r for *_, r in labels})
        m = np.zeros((len(roles), n), bool)
        for s, e, r in labels:
            m[roles.index(r), int(round(s / frame)):int(round(e / frame))] = True
        return m[:, uem]

    R, H = grid(ref), grid(hyp)
    co = R.astype(np.int64) @ H.T.astype(np.int64)
    ri, hi = linear_sum_assignment(-co)
    nref, nhyp = R.sum(0), H.sum(0)
    corr = sum((R[a] & H[b]).astype(np.int64) for a, b in zip(ri, hi))
    total = nref.sum()
    fa = np.maximum(0, nhyp - nref).sum()
    miss = np.maximum(0, nref - nhyp).sum()
    conf = (np.minimum(nref, nhyp) - corr).sum()
    return tuple(float(round(100 * v / total, 1)) for v in (fa, miss, conf, fa + miss + conf))


def quat_to_R(x, y, z, w):
    q = np.array([w, x, y, z], float)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def build_phases(raw, out):
    src = raw / "mateus_code" / "bifocal-or-tecno" / "annotations"
    total = [0] * 5
    all_rows = []
    for r in RUNS:
        rows = list(csv.DictReader(open(src / f"annotations_{r}.csv", encoding="utf-8")))
        segs = [(int(x["phase_id"]), x["phase_name"], float(x["start_sec"]), float(x["end_sec"])) for x in rows]
        ok_order = [s[0] for s in segs] == list(range(5)) and [s[1] for s in segs] == PHASES
        ok_cont = all(abs(segs[i][3] - segs[i + 1][2]) < 1e-9 for i in range(4))
        check(ok_order and ok_cont, f"Run {r}: five phases in order, contiguous, {segs[0][2]:.0f}-{segs[-1][3]:.0f} s")
        out_rows = []
        for pid, name, s, e in segs:
            total[pid] += e - s
            out_rows.append([pid, name, f"{s:.0f}", f"{e:.0f}", f"{e - s:.0f}"])
            all_rows.append([f"run{r:02d}", pid, name, f"{s:.0f}", f"{e:.0f}", f"{e - s:.0f}"])
        write_csv(out / "annotations" / "phases" / f"run{r:02d}.csv",
                  ["phase_id", "phase", "start_s", "end_s", "duration_s"], out_rows)
        ratio = sum(e - s for _, _, s, e in segs) / sum(PAPER_TABLE_II[r])
        check(1.0 < ratio < 1.06, f"Run {r}: annotated seconds / Table II frames = {ratio:.3f} "
                                  "(annotated vs. exported frames, expected ~1.03)")
    write_csv(out / "annotations" / "phases" / "all_runs.csv",
              ["run", "phase_id", "phase", "start_s", "end_s", "duration_s"], all_rows)
    s = sum(total)
    shares = [round(100 * t / s, 1) for t in total]
    check(round(s) == 19084, f"Total labeled seconds = {s:.0f} (paper: 19,084)")
    check(shares == [10.2, 16.8, 24.2, 36.1, 12.7], f"Phase shares = {shares} (paper: 10.2/16.8/24.2/36.1/12.7)")
    return {r: None for r in RUNS}


def write_speaker_files(labels, base, run, extra=None):
    """labels: list of (start, end, role); extra: list of dicts for additional CSV columns."""
    base.parent.mkdir(parents=True, exist_ok=True)
    order = sorted(range(len(labels)), key=lambda i: (labels[i][0], labels[i][1]))
    with open(base.with_suffix(".rttm"), "w", encoding="utf-8") as fh:
        for i in order:
            s, e, role = labels[i]
            fh.write(f"SPEAKER run{run:02d} 1 {s:.3f} {e - s:.3f} <NA> <NA> {role} <NA> <NA>\n")
    header = ["start_s", "end_s", "duration_s", "speaker"]
    extra_cols = ["mean_contrast_db", "merged_count", "padded_ms"] if extra else []
    rows = []
    for i in order:
        s, e, role = labels[i]
        row = [f"{s:.3f}", f"{e:.3f}", f"{e - s:.3f}", role]
        if extra:
            x = extra[i]
            val = x.get("mean_contrast_db", "")
            row += ["" if val in ("", "nan", "-inf", "inf") else val, x.get("merged_count", ""), x.get("padded_ms", "")]
        rows.append(row)
    write_csv(base.with_suffix(".csv"), header + extra_cols, rows)


def build_speakers(raw, out, channels):
    for r in AUDITED:
        ref = parse_audacity(raw / "speakers_audited" / f"Run_{r}_Labels_GroundTruth.txt")
        roles = {x[2] for x in ref}
        check(roles <= set(ROLES), f"Audited Run {r}: roles {sorted(roles)}")
        write_speaker_files(ref, out / "annotations" / "speakers" / "audited" / f"run{r:02d}", r)
    for r in RUNS:
        txt = parse_audacity(raw / "speakers_pseudo" / f"Run_{r}_AntiFragLabels.txt")
        csv_rows = list(csv.DictReader(open(raw / "speakers_pseudo" / f"Run_{r}_AntiFragLabels.csv", encoding="utf-8")))
        same = len(csv_rows) == len(txt) and all(
            abs(float(c["start_s"]) - t[0]) < 1e-3 and abs(float(c["end_s"]) - t[1]) < 1e-3 and c["role"] == t[2]
            for c, t in zip(csv_rows, txt))
        check(same, f"Pseudo Run {r}: .txt and .csv agree ({len(txt)} segments)")
        roles = {x[2] for x in txt}
        dead = {role for role, st in channels[r].items() if st == "dead"}
        absent = {role for role, st in channels[r].items() if st == "none"}
        check(not (roles & (dead | absent)), f"Pseudo Run {r}: no labels for dead/absent channels {sorted(dead | absent) or '-'}")
        write_speaker_files(txt, out / "annotations" / "speakers" / "pseudo" / f"run{r:02d}", r,
                            extra=csv_rows if same else None)
    for r in AUDITED:
        ref = parse_audacity(raw / "speakers_audited" / f"Run_{r}_Labels_GroundTruth.txt")
        hyp = parse_audacity(raw / "speakers_pseudo" / f"Run_{r}_AntiFragLabels.txt")
        got = der_collar(ref, hyp, 0.25, ignore=("Anesthesiologist",) if r == 3 else ())
        exp = PAPER_TABLE_V[r]
        check(all(abs(a - b) <= 0.15 for a, b in zip(got, exp)),
              f"Run {r} wearable DER reproduced: FA/Miss/Conf/DER = {got} (Table V: {exp})")


def build_channels(raw, out):
    wb = openpyxl.load_workbook(raw / "docs" / "Microphones_Missing_Mappings.xlsx", data_only=True)
    ws = wb.worksheets[0]
    header = [str(c).strip() if c else "" for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    channels = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        m = re.search(r"(\d+)", str(row[0]))
        if not m:
            continue
        r = int(m.group(1))
        st = {}
        for role in ["Room"] + ROLES[:-1]:
            v = row[header.index(role)] if role in header else None
            st[role] = "dead" if (v and str(v).strip().lower() == "no") else "ok"
        comment = str(row[header.index("Comments")] or "") if "Comments" in header else ""
        st["Spectator"] = "ok" if "spectator" in comment.lower() else "none"
        channels[r] = st
    rows = [[f"run{r:02d}"] + [channels[r][x] for x in ["Room"] + ROLES] for r in RUNS]
    write_csv(out / "metadata" / "audio_channels.csv",
              ["run", "room", "surgeon", "ota", "runner", "anesthesiologist", "nurse", "spectator"], rows)
    expected_dead = {1: {"Surgeon", "Anesthesiologist"}, 3: {"Anesthesiologist"}, 9: {"Surgeon"}}
    got_dead = {r: {k for k, v in channels[r].items() if v == "dead"} for r in RUNS}
    check(all(got_dead[r] == expected_dead.get(r, set()) for r in RUNS),
          "Dead wearable channels match paper Table III (Run 1 Surgeon+Anesth, Run 3 Anesth, Run 9 Surgeon)")
    check([r for r in RUNS if channels[r]["Spectator"] == "ok"] == [8], "Spectator channel only in Run 8")
    return channels


def build_calibration(raw, out):
    cal = out / "calibration"
    (cal / "intrinsics").mkdir(parents=True, exist_ok=True)
    (cal / "extrinsics").mkdir(parents=True, exist_ok=True)
    opencv = {"_doc": ("Factory intrinsics of the five Orbbec Femto Mega cameras, converted from intrinsics/camera0X.json. "
                       "K = [[fx,0,cx],[0,fy,cy],[0,0,1]] in pixels. dist = OpenCV order "
                       "[k1, k2, p1, p2, k3, k4, k5, k6] (rational model, cv2.CALIB_RATIONAL_MODEL). "
                       "color_to_depth: rotation (3x3) and translation (metres) of the color-depth pair, "
                       "named as in the recording software ('color2depth_transform'), quaternion order x, y, z, w.")}
    for c in range(1, 6):
        name = f"camera0{c}"
        src = raw / "calibration" / f"{name}.json"
        shutil.copy2(src, cal / "intrinsics" / f"{name}.json")
        d = json.load(open(src, encoding="utf-8"))["value0"]
        entry = {}
        for sensor in ("color", "depth"):
            p = d[f"{sensor}_parameters"]
            rd, td = p["radial_distortion"], p["tangential_distortion"]
            k = [rd[f"m{i}0"] for i in range(6)]
            dist = [k[0], k[1], td["m00"], td["m10"], k[2], k[3], k[4], k[5]]
            fx, fy, cx, cy = p["fov_x"], p["fov_y"], p["c_x"], p["c_y"]
            entry[sensor] = {"width": p["width"], "height": p["height"],
                             "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                             "dist": dist, "metric_radius": p.get("metric_radius")}
            ok = (0 < cx < p["width"] and 0 < cy < p["height"] and abs(fx - fy) / fx < 0.01)
            check(ok, f"{name} {sensor}: {p['width']}x{p['height']}, fx={fx:.1f}, fy={fy:.1f}, "
                      f"cx={cx:.1f}, cy={cy:.1f}")
        t = d["color2depth_transform"]
        q = t["rotation"]
        R = quat_to_R(q["x"], q["y"], q["z"], q["w"])
        tr = [t["translation"][f"m{i}0"] for i in range(3)]
        entry["color_to_depth"] = {"R": R, "t": tr, "quaternion_xyzw": [q["x"], q["y"], q["z"], q["w"]]}
        check(0.02 < np.linalg.norm(tr) < 0.05, f"{name} color-depth baseline {1000 * np.linalg.norm(tr):.1f} mm")
        opencv[name] = entry
    json.dump(opencv, open(cal / "intrinsics_opencv.json", "w", encoding="utf-8"), indent=1)

    docs = {
        "charuco_dist": ("charuco_dist_extrinsics.json", 1.6,
                         "Re-estimated by distortion-aware bundle adjustment on ChArUco corner detections "
                         "(every frame used for fitting; RMS reprojection error 1.6 px on held-out frames). Recommended default."),
        "after": ("after_extrinsics.json", 2.7,
                  "Calibration repeated after the recording sessions (setup calibration, extrinsics only). RMS 2.7 px."),
        "shipped": ("shipped_extrinsics.json", 4.3,
                    "Initial calibration delivered with the recording setup. RMS 4.3 px."),
    }
    for key, (fname, rms, text) in docs.items():
        d = json.load(open(raw / "calibration" / fname, encoding="utf-8"))
        res = {"_doc": text + " Convention: camera01 color camera = world frame; "
                              "p_world = R @ p_color + t; metres; OpenCV camera axes (x right, y down, z forward). "
                              "Use with the factory intrinsics and distortion in intrinsics_opencv.json.",
               "_rms_reprojection_px": rms, "_original_note": d.get("_doc", "")}
        for cam in [k for k in d if not k.startswith("_")]:
            R = np.array(d[cam]["R"])
            okR = np.abs(R.T @ R - np.eye(3)).max() < 1e-9 and abs(np.linalg.det(R) - 1) < 1e-9
            check(okR, f"extrinsics {key} {cam}: rotation orthonormal, |t| = {np.linalg.norm(d[cam]['t']):.3f} m")
            res[cam] = {"R": d[cam]["R"], "t": d[cam]["t"]}
        json.dump(res, open(cal / "extrinsics" / f"{key}.json", "w", encoding="utf-8"), indent=1)

    board = json.load(open(raw / "docs" / "aruco_config.json", encoding="utf-8")).get("charuco_board", {})
    return board


def build_docs(raw, out):
    meta = out / "metadata"
    wb = openpyxl.load_workbook(raw / "docs" / "ArUco_Mappings.xlsx", data_only=True)
    ws = wb.worksheets[0]
    rows, section = [], None
    for row in ws.iter_rows(values_only=True):
        cells = [c for c in row if c is not None]
        if not cells:
            continue
        first = str(cells[0]).strip()
        if first in ("ArUco Mappings",):
            continue
        if first == "Device":
            section = "object"
            continue
        if first == "Role":
            section = "role"
            continue
        ids = [int(c) for c in cells[1:] if str(c).strip().isdigit()]
        if section == "object":
            rows.append(["object", first, "marker_1;marker_2", ";".join(map(str, ids))])
        elif section == "role":
            rows.append(["staff", first, "head;front;back", ";".join(map(str, ids))])
    write_csv(meta / "aruco_markers.csv", ["type", "name", "positions", "marker_ids"], rows)
    all_ids = [int(i) for r in rows for i in r[3].split(";")]
    check(len(all_ids) == len(set(all_ids)) == 23, f"ArUco: {len(all_ids)} unique marker IDs (5 roles x 3 + 4 objects x 2)")

    shutil.copy2(raw / "docs" / "Workflow_Clean.xlsx", meta / "workflow_playbook.xlsx")
    wb = openpyxl.load_workbook(raw / "docs" / "Workflow_Clean.xlsx", data_only=True)
    ws = wb.worksheets[0]
    pb = [["" if v is None else str(v).strip() for v in row] for row in ws.iter_rows(values_only=True)
          if any(v is not None for v in row)]
    write_csv(meta / "workflow_playbook.csv", pb[0], pb[1:])

    wb = openpyxl.load_workbook(raw / "docs" / "Director_annotations.xlsx", data_only=True)
    events = []
    for ws in wb.worksheets:
        m = re.match(r"Con_Take\s*(\d+)", ws.title)
        if not m or int(m.group(1)) not in RUNS:
            continue
        r = int(m.group(1))
        lights_seen = 0
        for row in ws.iter_rows(values_only=True):
            if not row or row[0] is None:
                continue
            label = str(row[0])
            for key, ev in MILESTONES:
                if label.startswith(key):
                    val = "" if row[1] is None else str(row[1]).strip()
                    tm = re.search(r"(\d{1,2}:\d{2}(:\d{2})?)", val)
                    rest = re.sub(r"\d{1,2}:\d{2}(:\d{2})?|PM|AM", "", val).strip(" -()")
                    note = ""
                    if "forgot" in rest.lower() or "around" in rest.lower() or "ca." in str(row[2] if len(row) > 2 and row[2] else ""):
                        note = "approximate" if tm else "not logged"
                    if ev.startswith("lights"):
                        lights_seen += 1
                    events.append([f"run{r:02d}", ev, tm.group(1) if tm else "", note])
        times = {e[1]: e[2] for e in events if e[0] == f"run{r:02d}"}
        if times.get("last_person_out") and times.get("patient_out") and times["last_person_out"] < times["patient_out"]:
            for e in events:
                if e[0] == f"run{r:02d}" and e[1] == "last_person_out":
                    e[3] = "as logged (before patient_out)"
        for note in RUN_NOTES[r]:
            events.append([f"run{r:02d}", "note", "", note])
    write_csv(meta / "events.csv", ["run", "event", "director_time", "note"], events)
    check(len({e[0] for e in events}) == 9, "events.csv covers Runs 1-9 (director log, curated notes)")


def build_runs(out, phase_info, channels):
    rows = []
    for r in RUNS:
        folder = RUN_FOLDERS[r]
        m = re.match(r"rec_(\d{2})-(\d{2})-(\d{4})_(\d{2})-(\d{2})_Run_(\d+)", folder)
        date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
        utc = f"{m.group(4)}:{m.group(5)}"
        ph = list(csv.DictReader(open(out / "annotations" / "phases" / f"run{r:02d}.csv", encoding="utf-8")))
        dead = [k for k, v in channels[r].items() if v == "dead"]
        rows.append([f"run{r:02d}", folder, f"Con_Take {r}", date, utc,
                     "porcine (ex vivo)" if r == 9 else "plastic phantom",
                     "B (held-out test)" if r == 9 else "A (LOO-CV)",
                     ph[-1]["end_s"], sum(int(x["duration_s"]) for x in ph),
                     "audited + pseudo" if r in AUDITED else "pseudo",
                     ";".join(dead) or "-", " | ".join(RUN_NOTES[r])])
    write_csv(out / "metadata" / "runs.csv",
              ["run", "recording_folder", "take", "date", "folder_time", "tissue", "split",
               "phase_end_s", "labeled_s", "speaker_labels", "dead_wearables", "notes"], rows)


def build_sync_template(out):
    """offset_s = time on the room-camera timeline at which the stream starts;
    rate = stream seconds per timeline second (stream time = (t - offset_s) * rate)."""
    rows = []
    for r in RUNS:
        rows.append([f"run{r:02d}", "audio", "", "1.0", "to determine", "", "synced WAVs (audio/)"])
        parts = ["microscope_part1", "microscope_part2"] if r == 1 else ["microscope"]
        for s in parts:
            rows.append([f"run{r:02d}", s, "", "", "to determine", "", "microscope video (microscope/)"])
    write_csv(out / "sync" / "offsets.csv", ["run", "stream", "offset_s", "rate", "status", "method", "note"], rows)


def build_pyannote(raw, sync_dir, out):
    """Far-field pyannote outputs (Runs 3, 4, 7) and pseudo-label diagnostics from the sync files."""
    dvr = sync_dir / "Ophthalmo_4D_DVR_Sync"
    if not dvr.is_dir():
        return
    for r in AUDITED:
        hits = sorted(dvr.glob(f"Run_{r}/Run_{r}_*pyannote.txt"))
        if not hits:
            check(False, f"pyannote output for Run {r} not in the sync files")
            continue
        hyp = parse_audacity(hits[0])
        write_speaker_files(hyp, out / "annotations" / "baselines" / "pyannote_ambient" / f"run{r:02d}", r)
        ref = parse_audacity(raw / "speakers_audited" / f"Run_{r}_Labels_GroundTruth.txt")
        gt_drive = sync_dir_ref(dvr, r)
        if gt_drive is not None:
            check(parse_audacity(gt_drive) == ref, f"Run {r}: audited labels on the drive = released audited labels")
        if ref:
            got = der_collar(ref, hyp, 0.25, ignore=("Anesthesiologist",) if r == 3 else ())
            exp = PAPER_TABLE_V_AMBIENT[r]
            check(all(abs(a - b) <= 0.15 for a, b in zip(got, exp)),
                  f"Run {r} ambient DER reproduced: FA/Miss/Conf/DER = {got} (Table V: {exp})")
    diag = out / "annotations" / "speakers" / "pseudo" / "diagnostics"
    for f in sorted(dvr.glob("Run_*/Run_*_diarize_summary.json")):
        m = re.search(r"Run_(\d+)_diarize_summary", f.name)
        if m:
            diag.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, diag / f"run{int(m.group(1)):02d}_summary.json")


def sync_dir_ref(dvr, r):
    p = dvr / f"Run_{r}" / f"Run_{r}_Labels_GroundTruth.txt"
    return p if p.exists() else None


def write_validation(out):
    lines = ["# Validation of the metadata layer", "",
             "Generated by `build_release_metadata.py`. PASS = checked automatically against the paper or "
             "against internal consistency; CHECK = needs a look.", ""]
    lines += [f"- **{s}** {t}" for s, t in checks]
    (out / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    sync_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    print("Phases");      info = build_phases(raw, out)
    print("Channels");    channels = build_channels(raw, out)
    print("Speakers");    build_speakers(raw, out, channels)
    print("Calibration"); board = build_calibration(raw, out)
    print("Docs");        build_docs(raw, out)
    print("Runs");        build_runs(out, info, channels)
    build_sync_template(out)
    if sync_dir:
        print("Baselines"); build_pyannote(raw, sync_dir, out)
    json.dump(board, open(out / "calibration" / "charuco_board.json", "w", encoding="utf-8"), indent=1)
    docs = Path(__file__).resolve().parent / "release_docs"
    if docs.is_dir():  # README.md, LICENSE.md, calibration/README.md
        shutil.copytree(docs, out, dirs_exist_ok=True)
    write_validation(out)
    n_fail = sum(1 for s, _ in checks if s != "PASS")
    print(f"\n{len(checks)} checks, {n_fail} to look at. Output: {out}")


if __name__ == "__main__":
    main()
