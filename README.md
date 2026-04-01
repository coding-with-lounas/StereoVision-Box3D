# StereoVision-Box3D

3-D reconstruction of boxes using stereo vision (SIFT + triangulation).

## Project structure

```
StereoVision-Box3D/
│
├── data/
│   ├── calibration/        # 15–20 chessboard photos for intrinsic calibration
│   └── scene/              # left.jpg and right.jpg of the target boxes
│
├── calibration_results/
│   └── camera_params.xml   # Saved intrinsic parameters (K, distortion)
│
├── output/
│   ├── matches/            # SIFT match visualisations
│   └── reconstruction/     # cloud.ply / cloud.txt point clouds
│
├── src/
│   ├── calibration.py      # Intrinsic camera calibration
│   ├── features.py         # SIFT detection & matching
│   ├── triangulation.py    # Essential matrix + 3-D triangulation
│   └── main.py             # Full pipeline orchestrator
│
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add calibration images
#    Copy 15–20 chessboard photos into  data/calibration/

# 3. Add scene images
#    Copy left.jpg and right.jpg into  data/scene/

# 4. Run the pipeline (from the project root)
python src/main.py
```

The pipeline will:
1. Calibrate the camera (or load `calibration_results/camera_params.xml` if it exists).
2. Detect SIFT keypoints in both scene images and match them.
3. Triangulate matched points and export the 3-D point cloud.

## Box dimensions (real-world reference)

| Box | Width (mm) | Height (mm) | Depth (mm) |
|-----|-----------|------------|-----------|
| A   | —         | —          | —         |
| B   | —         | —          | —         |

> Fill in the table with the measured dimensions of your boxes.

## Dependencies

| Package | Version |
|---------|---------|
| opencv-python | ≥ 4.8.0 |
| numpy | ≥ 1.24.0 |