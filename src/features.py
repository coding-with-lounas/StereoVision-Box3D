"""
features.py
-----------
Detects SIFT keypoints in the left and right scene images,
matches them with a ratio test, and saves a visualisation to output/matches/.
"""

import os
import cv2
import numpy as np


# --- Configuration ---
LEFT_IMAGE_PATH = os.path.join("data", "scene", "left.jpg")
RIGHT_IMAGE_PATH = os.path.join("data", "scene", "right.jpg")
OUTPUT_MATCHES_DIR = os.path.join("output", "matches")
LOWE_RATIO = 0.75          # Lowe's ratio-test threshold
MIN_MATCH_COUNT = 8        # minimum good matches required


def load_scene_images(left_path: str = LEFT_IMAGE_PATH,
                      right_path: str = RIGHT_IMAGE_PATH):
    """Load the left and right scene images (BGR)."""
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None:
        raise FileNotFoundError(f"Left image not found: '{left_path}'")
    if right is None:
        raise FileNotFoundError(f"Right image not found: '{right_path}'")
    return left, right


def detect_and_match(left_img: np.ndarray,
                     right_img: np.ndarray,
                     ratio: float = LOWE_RATIO):
    """
    Detect SIFT features in both images and return good matches
    after Lowe's ratio test.

    Returns
    -------
    kp_left, kp_right : lists of cv2.KeyPoint
    good_matches      : list of cv2.DMatch (after ratio test)
    """
    sift = cv2.SIFT_create()

    gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    kp_left, des_left = sift.detectAndCompute(gray_left, None)
    kp_right, des_right = sift.detectAndCompute(gray_right, None)

    print(f"Keypoints — left: {len(kp_left)}, right: {len(kp_right)}")

    # BFMatcher with L2 norm, then ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_left, des_right, k=2)

    good_matches = [m for m, n in raw_matches if m.distance < ratio * n.distance]
    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < MIN_MATCH_COUNT:
        raise RuntimeError(
            f"Not enough good matches ({len(good_matches)} < {MIN_MATCH_COUNT}). "
            "Try different images or lower the ratio threshold."
        )

    return kp_left, kp_right, good_matches


def extract_matched_points(kp_left, kp_right, good_matches):
    """
    Extract (x, y) coordinates for each good match pair.

    Returns
    -------
    pts_left, pts_right : np.ndarray of shape (N, 2), dtype float32
    """
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])
    return pts_left, pts_right


def save_matches_visualization(left_img: np.ndarray,
                                right_img: np.ndarray,
                                kp_left, kp_right,
                                good_matches,
                                output_dir: str = OUTPUT_MATCHES_DIR,
                                filename: str = "sift_matches.jpg") -> str:
    """Draw and save a side-by-side match visualisation."""
    os.makedirs(output_dir, exist_ok=True)
    vis = cv2.drawMatches(
        left_img, kp_left,
        right_img, kp_right,
        good_matches[:50],   # draw at most 50 matches for clarity
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, vis)
    print(f"Match visualisation saved to '{out_path}'")
    return out_path


def run(left_path: str = LEFT_IMAGE_PATH,
        right_path: str = RIGHT_IMAGE_PATH) -> dict:
    """Full feature-detection and matching pipeline."""
    left_img, right_img = load_scene_images(left_path, right_path)
    kp_left, kp_right, good_matches = detect_and_match(left_img, right_img)
    pts_left, pts_right = extract_matched_points(kp_left, kp_right, good_matches)
    save_matches_visualization(left_img, right_img, kp_left, kp_right, good_matches)

    return {
        "left_img": left_img,
        "right_img": right_img,
        "kp_left": kp_left,
        "kp_right": kp_right,
        "good_matches": good_matches,
        "pts_left": pts_left,
        "pts_right": pts_right,
    }


if __name__ == "__main__":
    run()
