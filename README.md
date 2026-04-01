# StereoVision-Box3D

A Python pipeline that reconstructs the **3-D bounding boxes of physical
objects** from a pair of stereo photographs using classical computer-vision
techniques (camera calibration, SIFT feature matching, and triangulation).

---

## Project Structure

```
StereoVision_Project/
│
├── data/                       # Input data (your photos)
│   ├── calibration/            # 15-20 checkerboard calibration images
│   └── scene/                  # Stereo pair: left.jpg and right.jpg
│
├── calibration_results/        # Computed parameters (matrix K, distortion)
│   └── camera_params.xml       # Saved calibration for reuse
│
├── output/                     # Generated results
│   ├── matches/                # Visualisation of SIFT correspondences
│   └── reconstruction/         # Final point cloud (cloud.ply / cloud.txt)
│
├── src/                        # Python scripts
│   ├── calibration.py          # Compute intrinsic camera parameters
│   ├── features.py             # SIFT detection and matching
│   ├── triangulation.py        # 3-D coordinate computation
│   └── main.py                 # Main orchestration script
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your images

| Folder               | Content                                          |
|----------------------|--------------------------------------------------|
| `data/calibration/`  | 15–20 JPEG/PNG photos of a 9×6 checkerboard      |
| `data/scene/`        | Exactly two images: **left.jpg** and **right.jpg** |

### 3. Run the full pipeline

```bash
cd src
python main.py
```

Add `--recalibrate` to force re-computing camera parameters even when
`calibration_results/camera_params.xml` already exists.

### 4. Run individual steps

```bash
# Calibration only
python src/calibration.py

# Feature matching only (needs camera_params.xml)
python src/features.py

# Triangulation only (needs camera_params.xml + scene images)
python src/triangulation.py
```

---

## Configuration

Key constants that you may want to adjust are declared at the top of each
script:

| File              | Constant          | Default | Meaning                                 |
|-------------------|-------------------|---------|-----------------------------------------|
| `calibration.py`  | `CHECKERBOARD`    | (9, 6)  | Inner corners of the calibration board  |
| `calibration.py`  | `SQUARE_SIZE`     | 25.0    | Physical square size in **mm**          |
| `features.py`     | `RATIO_THRESHOLD` | 0.75    | Lowe's ratio-test threshold             |

---

## Real Box Dimensions (reference)

| Box | Width (mm) | Height (mm) | Depth (mm) |
|-----|-----------|-------------|------------|
| A   |           |             |            |
| B   |           |             |            |
| C   |           |             |            |

*(Fill in the measured dimensions of the boxes used during your experiment.)*

---

## Output Files

| File                                  | Description                             |
|---------------------------------------|-----------------------------------------|
| `calibration_results/camera_params.xml` | Camera matrix K and distortion coeffs  |
| `output/matches/matches.jpg`          | Visualisation of SIFT matches           |
| `output/reconstruction/cloud.ply`     | ASCII PLY point cloud                   |
| `output/reconstruction/cloud.txt`     | Space-separated X Y Z point cloud       |
