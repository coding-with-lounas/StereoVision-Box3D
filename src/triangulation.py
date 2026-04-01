"""
triangulation.py
----------------
Estimates the essential matrix, recovers the relative pose (R, t) between
the two cameras, and triangulates matched 2-D points into 3-D coordinates.
The resulting point cloud is saved to output/reconstruction/.
"""

import os
import cv2
import numpy as np


OUTPUT_RECONSTRUCTION_DIR = os.path.join("output", "reconstruction")


# ---------------------------------------------------------------------------
# Core triangulation
# ---------------------------------------------------------------------------

def compute_essential_matrix(pts_left: np.ndarray,
                              pts_right: np.ndarray,
                              camera_matrix: np.ndarray):
    """
    Compute the essential matrix from matched points and the intrinsic matrix.

    Returns
    -------
    E          : (3, 3) essential matrix
    mask       : inlier boolean mask (output of findEssentialMat)
    """
    E, mask = cv2.findEssentialMat(
        pts_left, pts_right, camera_matrix,
        method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    inliers = int(mask.sum())
    print(f"Essential matrix — inliers: {inliers}/{len(pts_left)}")
    return E, mask


def recover_pose(E: np.ndarray,
                 pts_left: np.ndarray,
                 pts_right: np.ndarray,
                 camera_matrix: np.ndarray,
                 mask: np.ndarray):
    """
    Recover the relative rotation R and translation t from E.

    Returns
    -------
    R, t  : rotation (3×3) and translation (3×1) of the right camera
            relative to the left camera
    pose_mask : inlier mask after cheirality check
    """
    _, R, t, pose_mask = cv2.recoverPose(E, pts_left, pts_right,
                                          camera_matrix, mask=mask)
    print(f"Recovered pose — inliers after cheirality: {int(pose_mask.sum())}")
    return R, t, pose_mask


def triangulate_points(pts_left: np.ndarray,
                       pts_right: np.ndarray,
                       R: np.ndarray,
                       t: np.ndarray,
                       camera_matrix: np.ndarray,
                       mask: np.ndarray) -> np.ndarray:
    """
    Triangulate inlier matched points into 3-D coordinates.

    Returns
    -------
    points_3d : (N, 3) array of 3-D points
    """
    inlier_mask = mask.ravel().astype(bool)
    pts_l = pts_left[inlier_mask]
    pts_r = pts_right[inlier_mask]

    # Projection matrices: P_left = K [I | 0], P_right = K [R | t]
    P_left = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = camera_matrix @ np.hstack((R, t))

    pts_4d = cv2.triangulatePoints(P_left, P_right,
                                    pts_l.T, pts_r.T)
    # Convert from homogeneous to 3-D
    points_3d = (pts_4d[:3] / pts_4d[3]).T
    print(f"Triangulated {len(points_3d)} 3-D point(s)")
    return points_3d


# ---------------------------------------------------------------------------
# Point-cloud I/O
# ---------------------------------------------------------------------------

def save_point_cloud_ply(points_3d: np.ndarray,
                          output_dir: str = OUTPUT_RECONSTRUCTION_DIR,
                          filename: str = "cloud.ply") -> str:
    """Save a point cloud to a PLY file (ASCII)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points_3d)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        for x, y, z in points_3d:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Point cloud saved to '{out_path}'")
    return out_path


def save_point_cloud_txt(points_3d: np.ndarray,
                          output_dir: str = OUTPUT_RECONSTRUCTION_DIR,
                          filename: str = "cloud.txt") -> str:
    """Save a point cloud to a plain-text file (x y z per line)."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    np.savetxt(out_path, points_3d, fmt="%.6f", header="x y z")
    print(f"Point cloud saved to '{out_path}'")
    return out_path


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run(pts_left: np.ndarray,
        pts_right: np.ndarray,
        camera_matrix: np.ndarray) -> np.ndarray:
    """
    Full triangulation pipeline:
      1. Estimate essential matrix
      2. Recover pose (R, t)
      3. Triangulate matched points
      4. Save point cloud (.ply and .txt)

    Returns
    -------
    points_3d : (N, 3) array of reconstructed 3-D points
    """
    E, mask = compute_essential_matrix(pts_left, pts_right, camera_matrix)
    R, t, pose_mask = recover_pose(E, pts_left, pts_right, camera_matrix, mask)
    points_3d = triangulate_points(pts_left, pts_right, R, t,
                                    camera_matrix, pose_mask)
    save_point_cloud_ply(points_3d)
    save_point_cloud_txt(points_3d)
    return points_3d


if __name__ == "__main__":
    # Quick smoke-test with dummy data
    print("Run main.py to execute the full pipeline.")
