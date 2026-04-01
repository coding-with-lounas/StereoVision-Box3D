"""
triangulation.py
----------------
Estimate the Essential matrix from matched point pairs, recover the camera
pose (R, t) between the two views, and triangulate the 3-D coordinates of
the matched points.  The resulting point cloud is exported as a PLY file.

Usage (standalone):
    python triangulation.py
    (requires calibration_results/camera_params.xml and matched points)

The module can also be called programmatically – see triangulate().
"""

import os

import cv2
import numpy as np

from calibration import load_params
from features import detect_and_match

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "reconstruction")
OUTPUT_PLY = os.path.join(OUTPUT_DIR, "cloud.ply")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "cloud.txt")


def triangulate(
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_ply: str = OUTPUT_PLY,
    output_txt: str = OUTPUT_TXT,
) -> np.ndarray:
    """
    Triangulate 3-D points from two sets of matched 2-D image points.

    Steps
    -----
    1. Undistort matched 2-D points.
    2. Compute the Essential matrix (RANSAC).
    3. Recover relative camera pose (R, t).
    4. Triangulate via cv2.triangulatePoints.
    5. Save the point cloud to PLY and TXT files.

    Parameters
    ----------
    pts_left, pts_right : np.ndarray, shape (N, 2)
        Matched image coordinates in the left and right views.
    camera_matrix : np.ndarray, shape (3, 3)
    dist_coeffs   : np.ndarray
    output_ply    : str   – destination for the PLY point cloud
    output_txt    : str   – destination for the TXT point cloud

    Returns
    -------
    points_3d : np.ndarray, shape (N, 3)
    """
    # 1. Undistort points
    pts_left_u = cv2.undistortPoints(
        pts_left.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(-1, 2)
    pts_right_u = cv2.undistortPoints(
        pts_right.reshape(-1, 1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    ).reshape(-1, 2)

    # 2. Essential matrix
    E, mask = cv2.findEssentialMat(
        pts_left_u, pts_right_u, camera_matrix,
        method=cv2.RANSAC, prob=0.999, threshold=1.0,
    )
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")

    inlier_mask = mask.ravel().astype(bool)
    pts_left_in = pts_left_u[inlier_mask]
    pts_right_in = pts_right_u[inlier_mask]
    print(f"Essential matrix inliers: {inlier_mask.sum()} / {len(inlier_mask)}")

    # 3. Recover pose
    _, R, t, pose_mask = cv2.recoverPose(E, pts_left_in, pts_right_in, camera_matrix)

    # 4. Triangulate
    P1 = camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = camera_matrix @ np.hstack([R, t])

    pts_4d = cv2.triangulatePoints(P1, P2, pts_left_in.T, pts_right_in.T)
    pts_4d /= pts_4d[3]          # homogeneous → Cartesian
    points_3d = pts_4d[:3].T     # shape (N, 3)

    # Filter points behind camera (negative depth)
    front_mask = points_3d[:, 2] > 0
    points_3d = points_3d[front_mask]
    print(f"Triangulated {len(points_3d)} 3-D points (positive depth).")

    # 5. Save
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    _save_ply(points_3d, output_ply)
    _save_txt(points_3d, output_txt)

    return points_3d


def _save_ply(points_3d: np.ndarray, filepath: str):
    """Save a point cloud as an ASCII PLY file."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points_3d)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(header)
        for pt in points_3d:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
    print(f"Point cloud (PLY) saved to '{filepath}'.")


def _save_txt(points_3d: np.ndarray, filepath: str):
    """Save a point cloud as a plain space-separated text file (X Y Z per line)."""
    np.savetxt(filepath, points_3d, fmt="%.6f", header="X Y Z")
    print(f"Point cloud (TXT) saved to '{filepath}'.")


if __name__ == "__main__":
    camera_matrix, dist_coeffs = load_params()
    pts_left, pts_right, _, _ = detect_and_match()
    triangulate(pts_left, pts_right, camera_matrix, dist_coeffs)
