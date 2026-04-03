# StereoVision — 3D Reconstruction from Stereo Images

> A lightweight stereo vision pipeline that reconstructs 3D point clouds from two calibrated camera views using SIFT feature matching and triangulation.

---

## Overview

This project implements a complete stereo vision system:

1. **Camera Calibration** — Estimate intrinsic parameters from a checkerboard pattern
2. **Feature Detection & Matching** — Detect and match SIFT keypoints across stereo image pairs
3. **3D Reconstruction** — Triangulate matched points to recover real-world coordinates
4. **Point Cloud Visualization** — Export and visualize the reconstructed 3D scene

---

## Scene Configuration

Three boxes of known dimensions are arranged on a flat surface and photographed from two laterally translated camera positions (baseline **b**).

| Object | Length (cm) | Width (cm) | Height (cm) |
|--------|------------|-----------|------------|
| Box 1 (large) | ___ | ___ | ___ |
| Box 2 (medium) | ___ | ___ | ___ |
| Box 3 (small) | ___ | ___ | ___ |

| Acquisition Parameter | Value |
|----------------------|-------|
| Camera / Device | GALAXY M36 |
| Image Resolution | 3060 × 4080 px |
| Baseline b | 10 cm |
| Checkerboard square size | 2.5 cm |
| Number of calibration images | 17 |

---

## Project Structure

```
StereoVision_Project/
│
├── data/
│   ├── calibration/            # Checkerboard images (15–20 shots)
│   └── scene/
│       ├── left.jpg            # Left view
│       └── right.jpg           # Right view
│
├── calibration_results/
│   └── camera_params.xml       # Intrinsic matrix K + distortion coefficients
│
├── output/
│   ├── matches/                # SIFT correspondence visualizations
│   └── reconstruction/         # Output point cloud (cloud.ply)
│
├── src/
│   ├── calibration.py          # Intrinsic parameter estimation
│   ├── features.py             # SIFT detection and matching
│   ├── triangulation.py        # 3D coordinate computation
│   └── main.py                 # End-to-end pipeline
│
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
opencv-python
opencv-contrib-python
numpy
open3d
matplotlib
```

---

## Usage

Run each step individually or use the full pipeline:

```bash
# Step 1 — Camera calibration
python src/calibration.py

# Step 2 — SIFT detection and matching
python src/features.py

# Step 3 — 3D triangulation
python src/triangulation.py

# Or run everything at once
python src/main.py
```

---

## Method

### Camera Model

The intrinsic matrix **K** maps 3D camera coordinates to 2D image coordinates:

```
K = | f   0   cx |
    | 0   f   cy |
    | 0   0   1  |
```

### Depth Recovery via Triangulation

Given a stereo pair with known baseline **b** and focal length **f**:

```
Z = (f × b) / disparity        where disparity = u_left − u_right
X = (u_left − cx) × Z / f
Y = (v_left − cy) × Z / f
```

A larger disparity → object is **closer** to the camera.

---

## Results

- Dense or semi-dense 3D point cloud of the scene
- Reconstructed box dimensions validated against ground truth measurements
- Point cloud exported as `.ply` (compatible with MeshLab, Open3D, CloudCompare)

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-contrib-python` | SIFT, calibration, image I/O |
| `numpy` | Linear algebra, triangulation |
| `open3d` | Point cloud visualization |
| `matplotlib` | 2D plots and match visualization |

---

## Author
| **Name** | IDJOURDIKENE LOUNAS |
| **Year** | 2025/2026 |
