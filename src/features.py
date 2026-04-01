"""
features.py
-----------
Detect SIFT keypoints in the left and right stereo images and match them
using a ratio test (Lowe, 2004).  Optionally saves a visualisation of the
matches to output/matches/.

Usage (standalone):
    python features.py

Expected input:  data/scene/left.jpg  and  data/scene/right.jpg
Output:          output/matches/matches.jpg  (optional visualisation)
"""

import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "scene")
MATCHES_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "matches")

LEFT_IMAGE = os.path.join(SCENE_DIR, "left.jpg")
RIGHT_IMAGE = os.path.join(SCENE_DIR, "right.jpg")
MATCHES_OUTPUT = os.path.join(MATCHES_DIR, "matches.jpg")

RATIO_THRESHOLD = 0.75   # Lowe's ratio test threshold


def detect_and_match(
    left_path: str = LEFT_IMAGE,
    right_path: str = RIGHT_IMAGE,
    ratio_threshold: float = RATIO_THRESHOLD,
    save_visualization: bool = True,
    output_path: str = MATCHES_OUTPUT,
) -> tuple:
    """
    Detect SIFT features in both images and match them with a ratio test.

    Parameters
    ----------
    left_path, right_path : str
        Paths to the left and right stereo images.
    ratio_threshold : float
        Lowe's ratio test threshold (default 0.75).
    save_visualization : bool
        When True, save a match visualisation to *output_path*.
    output_path : str
        Destination for the match visualisation image.

    Returns
    -------
    pts_left  : np.ndarray, shape (N, 2)  – matched 2-D points in left image
    pts_right : np.ndarray, shape (N, 2)  – matched 2-D points in right image
    img_left  : np.ndarray               – left image (BGR)
    img_right : np.ndarray               – right image (BGR)
    """
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None:
        raise FileNotFoundError(f"Left image not found: '{left_path}'")
    if img_right is None:
        raise FileNotFoundError(f"Right image not found: '{right_path}'")

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # SIFT detector (patent-free since OpenCV 4.4)
    sift = cv2.SIFT_create()
    kp_left, desc_left = sift.detectAndCompute(gray_left, None)
    kp_right, desc_right = sift.detectAndCompute(gray_right, None)

    print(f"Keypoints – left: {len(kp_left)}, right: {len(kp_right)}")

    if desc_left is None or desc_right is None or len(kp_left) == 0 or len(kp_right) == 0:
        raise RuntimeError("No SIFT descriptors found in one or both images.")

    # BFMatcher with L2 norm + k-NN (k=2) for ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc_left, desc_right, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    print(f"Good matches after ratio test ({ratio_threshold}): {len(good_matches)}")

    if len(good_matches) < 8:
        raise RuntimeError(
            f"Only {len(good_matches)} good matches found. "
            "Need at least 8 for reliable geometry estimation."
        )

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    if save_visualization:
        _save_match_visualization(
            img_left, kp_left, img_right, kp_right, good_matches, output_path
        )

    return pts_left, pts_right, img_left, img_right


def _save_match_visualization(img_left, kp_left, img_right, kp_right, matches, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vis = cv2.drawMatches(
        img_left, kp_left,
        img_right, kp_right,
        matches[:50],   # draw at most 50 matches for readability
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(output_path, vis)
    print(f"Match visualisation saved to '{output_path}'.")


if __name__ == "__main__":
    detect_and_match()
