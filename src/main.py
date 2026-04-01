"""
main.py
-------
Orchestrates the full StereoVision-Box3D pipeline:
  1. Camera calibration  (or load existing params)
  2. SIFT feature detection & matching
  3. 3-D triangulation & point-cloud export
"""

import os
import sys

# Make sure the src/ directory is on the path when run directly
sys.path.insert(0, os.path.dirname(__file__))

import calibration
import features
import triangulation

CALIBRATION_PARAMS_FILE = os.path.join("calibration_results", "camera_params.xml")


def main():
    print("=" * 60)
    print("  StereoVision-Box3D  —  3-D Reconstruction Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 – Camera calibration
    # ------------------------------------------------------------------
    print("\n[Step 1] Camera calibration")

    if os.path.isfile(CALIBRATION_PARAMS_FILE):
        print(f"Found existing calibration file: '{CALIBRATION_PARAMS_FILE}'")
        answer = input("Re-run calibration? [y/N] ").strip().lower()
        if answer == "y":
            params = calibration.calibrate(output_file=CALIBRATION_PARAMS_FILE)
        else:
            params = calibration.load_params(CALIBRATION_PARAMS_FILE)
            print("Calibration parameters loaded.")
    else:
        params = calibration.calibrate(output_file=CALIBRATION_PARAMS_FILE)

    camera_matrix = params["camera_matrix"]
    print(f"\nCamera matrix (K):\n{camera_matrix}")

    # ------------------------------------------------------------------
    # Step 2 – SIFT feature detection & matching
    # ------------------------------------------------------------------
    print("\n[Step 2] SIFT feature detection & matching")
    feat = features.run()
    pts_left = feat["pts_left"]
    pts_right = feat["pts_right"]

    # ------------------------------------------------------------------
    # Step 3 – 3-D triangulation
    # ------------------------------------------------------------------
    print("\n[Step 3] Triangulation")
    points_3d = triangulation.run(pts_left, pts_right, camera_matrix)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Reconstruction complete — {len(points_3d)} 3-D point(s) generated.")
    print(f"  Point cloud: output/reconstruction/cloud.ply")
    print("=" * 60)


if __name__ == "__main__":
    main()
