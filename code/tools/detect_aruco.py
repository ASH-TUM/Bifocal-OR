"""Bifocal-OR: detect the ArUco markers in the room-camera color frames.

Markers: OpenCV DICT_4X4_250, 10 x 10 cm.
- Staff roles wear three markers each (head / front / back): Surgeon 20-22, OTA 30-32,
  Anesthesiologist 40-42, Runner 50-52, Nurse 60-62.
- Mobile objects carry two: runner pushcart 100-101, patient bed 102-103,
  assistant pushcart 104-105, microscope 106-107.

The script writes one CSV row per detected marker:
    run, camera, frame, marker_id, role_or_object, part, x0, y0, x1, y1, x2, y2, x3, y3
with the four corners in pixels, in OpenCV order: top-left, top-right, bottom-right, bottom-left.
With --calibration it also writes the marker pose in that camera's color frame
(rvec, tvec in metres), estimated with the released intrinsics and distortion.

Input: a folder of color frames named color_<frame>_camera0X.jpg. Either extract the
release archive runs/runNN/room/color/camera0X.tar, or use the raw export_full/color folder.

Usage:
    python detect_aruco.py --frames <folder> --camera camera01 --run 3 --out run03_camera01_aruco.csv
    python detect_aruco.py --frames <folder> --camera camera01 --run 3 --step 8 \
        --calibration <release>/calibration/intrinsics_opencv.json --out run03_camera01_aruco.csv
    --step N   process every N-th frame (default 1 = all, about 7.5 frames per second)
Needs opencv-python >= 4.7 (cv2.aruco.ArucoDetector) and numpy.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

MARKER_SIZE_M = 0.10
ROLES = {20: "Surgeon", 30: "OTA", 40: "Anesthesiologist", 50: "Runner", 60: "Nurse"}
PARTS = {0: "head", 1: "front", 2: "back"}
OBJECTS = {100: "runner_pushcart", 102: "patient_bed", 104: "assistant_pushcart", 106: "microscope"}


def describe(marker_id):
    base = marker_id - marker_id % 10
    if base in ROLES and marker_id % 10 in PARTS:
        return ROLES[base], PARTS[marker_id % 10]
    base = marker_id - marker_id % 2
    if base in OBJECTS:
        return OBJECTS[base], f"marker{marker_id % 2 + 1}"
    return "unassigned", ""


def load_calibration(path, camera):
    """Camera matrix and distortion of the color camera from a calibration JSON. Works with the
    release file calibration/intrinsics_opencv.json (all cameras: camera0X -> color -> K, dist) and
    with single-camera files (keys searched loosely)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, dict) and camera in d:
        d = d[camera]
    if isinstance(d, dict) and isinstance(d.get("color"), dict):
        d = d["color"]

    def find(obj, names):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in names:
                    return v
                r = find(v, names)
                if r is not None:
                    return r
        return None

    K = find(d, {"camera_matrix", "k", "intrinsic_matrix", "intrinsics_matrix_3x3"})
    if K is None:
        fx, fy = find(d, {"fx"}), find(d, {"fy"})
        cx, cy = find(d, {"cx"}), find(d, {"cy"})
        if None in (fx, fy, cx, cy):
            sys.exit(f"No intrinsics found in {path}")
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    dist = find(d, {"dist_coeffs", "distortion", "distortion_coefficients", "dist"})
    if isinstance(dist, dict):
        dist = [dist.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
    return np.asarray(K, float).reshape(3, 3), np.asarray(dist if dist is not None else [0] * 5, float).ravel()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with color_<frame>_camera0X.jpg")
    ap.add_argument("--camera", required=True, help="camera01 ... camera05")
    ap.add_argument("--run", required=True, type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--calibration", help="release calibration JSON of this camera (optional, for poses)")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.frames, f"color_*_{a.camera}.jpg")))
    if not files:
        sys.exit(f"No color_*_{a.camera}.jpg in {a.frames}")
    frame_of = {f: int(re.search(r"color_(\d+)_", os.path.basename(f)).group(1)) for f in files}
    files = [f for f in files if frame_of[f] % a.step == 0]

    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
                                       cv2.aruco.DetectorParameters())
    K = dist = None
    if a.calibration:
        K, dist = load_calibration(a.calibration, a.camera)
        h = MARKER_SIZE_M / 2
        obj_pts = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

    n_det = 0
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["run", "camera", "frame", "marker_id", "role_or_object", "part"] + \
                 [f"{c}{i}" for i in range(4) for c in ("x", "y")]
        if K is not None:
            header += ["rvec_x", "rvec_y", "rvec_z", "tvec_x_m", "tvec_y_m", "tvec_z_m"]
        w.writerow(header)
        for i, f in enumerate(files):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            corners, ids, _ = detector.detectMarkers(img)
            if ids is None:
                continue
            for c, mid in zip(corners, ids.ravel()):
                pts = c.reshape(4, 2)
                who, part = describe(int(mid))
                row = [a.run, a.camera, frame_of[f], int(mid), who, part] + [f"{v:.1f}" for v in pts.ravel()]
                if K is not None:
                    ok, rvec, tvec = cv2.solvePnP(obj_pts, pts.astype(np.float32), K, dist,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    row += [f"{v:.4f}" for v in rvec.ravel()] + [f"{v:.4f}" for v in tvec.ravel()] if ok else [""] * 6
                w.writerow(row)
                n_det += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)} frames, {n_det} detections")
    print(f"{len(files)} frames processed, {n_det} marker detections -> {a.out}")


if __name__ == "__main__":
    main()
