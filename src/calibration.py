"""
calibration.py
--------------
Computes intrinsic camera parameters from a set of chessboard images.
Results are saved to calibration_results/camera_params.xml.
"""

import os
import cv2
import numpy as np
import glob


# --- Configuration ---
CHESSBOARD_SIZE = (9, 6)          # inner corners (columns, rows)
SQUARE_SIZE_MM = 25.0             # real-world square side in millimetres
CALIBRATION_IMAGES_DIR = os.path.join("data", "calibration")
OUTPUT_FILE = os.path.join("calibration_results", "camera_params.xml")


def find_chessboard_corners(images_dir: str, board_size: tuple):
    """Detect chessboard corners in all images inside *images_dir*."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points for a flat chessboard (z = 0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []   # 3-D world points
    img_points = []   # 2-D image points

    pattern = os.path.join(images_dir, "*.jpg")
    image_files = glob.glob(pattern)
    if not image_files:
        pattern = os.path.join(images_dir, "*.png")
        image_files = glob.glob(pattern)

    if not image_files:
        raise FileNotFoundError(
            f"No calibration images found in '{images_dir}'. "
            "Place .jpg or .png chessboard images there."
        )

    image_size = None
    for path in sorted(image_files):
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if found:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            print(f"[OK]  {os.path.basename(path)}")
        else:
            print(f"[--]  {os.path.basename(path)}  (corners not found)")

    if len(obj_points) < 4:
        raise RuntimeError(
            f"Only {len(obj_points)} valid image(s) found. "
            "Need at least 4 for a reliable calibration."
        )

    return obj_points, img_points, image_size


def calibrate(images_dir: str = CALIBRATION_IMAGES_DIR,
              board_size: tuple = CHESSBOARD_SIZE,
              output_file: str = OUTPUT_FILE) -> dict:
    """Run calibration and save the result to *output_file*."""
    print(f"\nSearching for calibration images in '{images_dir}' …")
    obj_points, img_points, image_size = find_chessboard_corners(images_dir, board_size)

    print(f"\nCalibrating with {len(obj_points)} image(s) …")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    mean_error = _reprojection_error(obj_points, img_points, rvecs, tvecs,
                                     camera_matrix, dist_coeffs)

    print(f"\nCamera matrix (K):\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
    print(f"Mean reprojection error: {mean_error:.4f} px")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    _save_params(output_file, camera_matrix, dist_coeffs, image_size)
    print(f"\nCalibration saved to '{output_file}'")

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "reprojection_error": mean_error,
    }


def load_params(filepath: str = OUTPUT_FILE) -> dict:
    """Load previously saved calibration parameters from an XML file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Calibration file '{filepath}' not found. Run calibration first."
        )
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    image_size = tuple(fs.getNode("image_size").mat().astype(int).flatten().tolist())
    fs.release()
    return {"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs,
            "image_size": image_size}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reprojection_error(obj_points, img_points, rvecs, tvecs,
                        camera_matrix, dist_coeffs) -> float:
    total_error = 0.0
    for i, objp in enumerate(obj_points):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        total_error += cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
    return total_error / len(obj_points)


def _save_params(filepath: str, camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray, image_size: tuple) -> None:
    fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", camera_matrix)
    fs.write("dist_coeffs", dist_coeffs)
    fs.write("image_size", np.array(image_size, dtype=np.int32))
    fs.release()


if __name__ == "__main__":
    calibrate()
