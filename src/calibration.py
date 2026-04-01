"""
calibration.py
--------------
Compute the intrinsic camera parameters (camera matrix K and distortion
coefficients) from a set of checkerboard calibration images and save the
results to an OpenCV XML file for later reuse.

Usage (standalone):
    python calibration.py

Expected input:  images in  data/calibration/  (*.jpg or *.png)
Output:          calibration_results/camera_params.xml
"""

import glob
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKERBOARD = (9, 6)          # inner corners per row × per column
SQUARE_SIZE = 25.0             # physical size of one square in mm

CALIBRATION_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "calibration")
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "calibration_results", "camera_params.xml"
)


def collect_object_points(checkerboard: tuple, square_size: float) -> np.ndarray:
    """Return the 3-D coordinates of inner corners for one checkerboard image."""
    rows, cols = checkerboard
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate(
    images_dir: str = CALIBRATION_DIR,
    checkerboard: tuple = CHECKERBOARD,
    square_size: float = SQUARE_SIZE,
    output_file: str = OUTPUT_FILE,
) -> tuple:
    """
    Run camera calibration on all images found in *images_dir*.

    Returns
    -------
    camera_matrix : np.ndarray, shape (3, 3)
    dist_coeffs   : np.ndarray, shape (1, 5) or similar
    """
    objp = collect_object_points(checkerboard, square_size)

    obj_points = []   # 3-D points in real-world space
    img_points = []   # 2-D points in image plane

    pattern = os.path.join(images_dir, "*.[jJ][pP][gG]")
    images = glob.glob(pattern)
    images += glob.glob(os.path.join(images_dir, "*.png"))
    images += glob.glob(os.path.join(images_dir, "*.PNG"))

    if not images:
        raise FileNotFoundError(
            f"No calibration images found in '{images_dir}'. "
            "Add JPEG or PNG checkerboard photos and re-run."
        )

    print(f"Found {len(images)} calibration image(s) in '{images_dir}'.")

    image_size = None
    for fname in sorted(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"  [WARN] Cannot read '{fname}', skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if ret:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"  [OK]   Checkerboard detected in '{os.path.basename(fname)}'.")
        else:
            print(f"  [SKIP] Checkerboard NOT detected in '{os.path.basename(fname)}'.")

    if len(obj_points) < 4:
        raise RuntimeError(
            f"Only {len(obj_points)} usable image(s) found. "
            "At least 4 are needed for reliable calibration."
        )

    print(f"\nCalibrating using {len(obj_points)} image(s)…")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    mean_error = _reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    print(f"Calibration RMS re-projection error: {ret:.4f} px")
    print(f"Mean re-projection error:             {mean_error:.4f} px")

    _save_params(output_file, camera_matrix, dist_coeffs, image_size)
    return camera_matrix, dist_coeffs


def _reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    for i, objp in enumerate(obj_points):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        total_error += error
    return total_error / len(obj_points)


def _save_params(output_file: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size: tuple):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fs = cv2.FileStorage(output_file, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.release()
    print(f"\nCalibration parameters saved to '{output_file}'.")


def load_params(params_file: str = OUTPUT_FILE) -> tuple:
    """
    Load previously saved calibration parameters.

    Returns
    -------
    camera_matrix : np.ndarray
    dist_coeffs   : np.ndarray
    """
    fs = cv2.FileStorage(params_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise RuntimeError(f"Failed to load calibration parameters from '{params_file}'.")
    return camera_matrix, dist_coeffs


if __name__ == "__main__":
    calibrate()
