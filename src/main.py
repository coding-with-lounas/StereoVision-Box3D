"""
main.py
-------
Main entry point for the StereoVision-Box3D pipeline.

Pipeline
--------
1. Calibration  – compute intrinsic camera parameters from checkerboard
                  images and save them to calibration_results/camera_params.xml.
                  If the file already exists the calibration step is skipped.
2. Features     – detect SIFT keypoints and match them between the left and
                  right scene images.
3. Triangulation– recover the stereo camera pose and triangulate 3-D points;
                  export the point cloud as output/reconstruction/cloud.ply
                  and output/reconstruction/cloud.txt.

Usage
-----
    python main.py [--recalibrate]

Options
-------
    --recalibrate   Force re-running calibration even if camera_params.xml exists.
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the Python path so that relative imports
# work regardless of the working directory from which main.py is called.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from calibration import OUTPUT_FILE as PARAMS_FILE
from calibration import calibrate, load_params
from features import detect_and_match
from triangulation import triangulate


def run_pipeline(recalibrate: bool = False) -> None:
    """Execute the full stereo-vision pipeline."""

    # ------------------------------------------------------------------
    # Step 1 – Calibration
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1 – Camera Calibration")
    print("=" * 60)

    if recalibrate or not os.path.isfile(PARAMS_FILE):
        camera_matrix, dist_coeffs = calibrate()
    else:
        print(f"Calibration file '{PARAMS_FILE}' already exists – loading parameters.")
        camera_matrix, dist_coeffs = load_params(PARAMS_FILE)

    print("\nCamera matrix (K):")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

    # ------------------------------------------------------------------
    # Step 2 – Feature Detection and Matching
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 – SIFT Feature Detection and Matching")
    print("=" * 60)

    pts_left, pts_right, img_left, img_right = detect_and_match(
        save_visualization=True
    )
    print(f"Matched point pairs: {len(pts_left)}")

    # ------------------------------------------------------------------
    # Step 3 – Triangulation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3 – 3-D Triangulation")
    print("=" * 60)

    points_3d = triangulate(pts_left, pts_right, camera_matrix, dist_coeffs)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  3-D points reconstructed : {len(points_3d)}")
    print(f"  Point cloud (PLY)        : output/reconstruction/cloud.ply")
    print(f"  Point cloud (TXT)        : output/reconstruction/cloud.txt")
    print(f"  Match visualisation      : output/matches/matches.jpg")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="StereoVision-Box3D: stereo reconstruction pipeline"
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force re-running camera calibration even if camera_params.xml already exists.",
    )
    args = parser.parse_args()
    run_pipeline(recalibrate=args.recalibrate)


if __name__ == "__main__":
    main()
