# Tools

| Script | What it does |
| --- | --- |
| `detect_aruco.py` | Detects the ArUco markers (OpenCV `DICT_4X4_250`, 10 × 10 cm) in the room-camera color frames and writes one CSV row per detection (run, camera, frame, marker ID, role or object, body part, four corners in pixels). With `--calibration` it also estimates each marker's pose in the camera frame. The marker assignment is in `metadata/aruco_markers.csv` of the dataset. |
| `blind_boundaries.py` | Blinded second annotation of the 36 internal phase boundaries (shuffled 3-min windows of all five cameras plus ambient audio, browser page for annotating) and the agreement score reported in the paper. |

```
pip install opencv-python numpy pillow

# markers of camera 1 in Run 3, every 8th frame (about 1 per second), with poses
tar -xf runs/run03/room/color/camera01.tar -C run03_color
python detect_aruco.py --frames run03_color --camera camera01 --run 3 --step 8 \
    --calibration calibration/intrinsics_opencv.json --out run03_camera01_aruco.csv
```

`--calibration` accepts a JSON file with a 3 × 3 camera matrix (`camera_matrix`, `K`, ...) or `fx`, `fy`, `cx`, `cy`, and a distortion vector. With `calibration/intrinsics_opencv.json` the script picks the entry of the camera given with `--camera`.

The usage of `blind_boundaries.py` is in its header.
