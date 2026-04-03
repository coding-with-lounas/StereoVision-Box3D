"""
features.py
-----------
SIFT keypoint detection, descriptor computation, and stereo matching.

Pipeline:
    1. Load left and right images
    2. Detect SIFT keypoints + compute descriptors
    3. Match descriptors using BFMatcher + Lowe's ratio test
    4. Visualize and save matches
    5. Return matched point coordinates for triangulation
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

LEFT_IMAGE_PATH  = "data/scene/left.jpg"
RIGHT_IMAGE_PATH = "data/scene/right.jpg"
OUTPUT_DIR       = "output/matches/"
LOWE_RATIO       = 0.7  # Lowe's ratio test threshold (standard = 0.75)

# ─────────────────────────────────────────────
#  STEP 1 — Load images
# ─────────────────────────────────────────────

def load_images(left_path, right_path):
    """Load left and right stereo images in grayscale."""
    img_left  = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        raise FileNotFoundError(
            f"Could not load images.\n"
            f"  Left  → {left_path}\n"
            f"  Right → {right_path}\n"
            f"Check that the paths are correct."
        )

    gray_left  = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    print(f"[✓] Images loaded: {img_left.shape[:2]} (H×W)")
    return img_left, img_right, gray_left, gray_right


# ─────────────────────────────────────────────
#  STEP 2 & 3 — Detect keypoints + descriptors
# ─────────────────────────────────────────────

def detect_and_compute(gray_left, gray_right):
    """
    Detect SIFT keypoints and compute descriptors for both images.

    SIFT finds points that are:
      - Stable across scales (scale-invariant)
      - Stable across rotations (rotation-invariant)
      - Described by a 128-dimensional vector
    """
    sift = cv2.SIFT_create()

    kp_left,  desc_left  = sift.detectAndCompute(gray_left,  None)
    kp_right, desc_right = sift.detectAndCompute(gray_right, None)

    print(f"[✓] Keypoints detected:")
    print(f"    Left  image → {len(kp_left)}  keypoints")
    print(f"    Right image → {len(kp_right)} keypoints")

    return kp_left, desc_left, kp_right, desc_right


# ─────────────────────────────────────────────
#  STEP 4 — Match descriptors (BFMatcher + Lowe)
# ─────────────────────────────────────────────

def match_descriptors(desc_left, desc_right, ratio=LOWE_RATIO):
    """
    Match SIFT descriptors using Brute-Force Matcher + Lowe's ratio test.

    Lowe's ratio test:
        For each keypoint, we find the 2 best matches (nearest neighbors).
        We keep the match only if:
            distance(best_match) < ratio × distance(second_best_match)
        This filters out ambiguous matches.
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)  # L2 norm for SIFT descriptors

    # k=2 → get the 2 nearest neighbors for each descriptor
    raw_matches = bf.knnMatch(desc_left, desc_right, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    print(f"[✓] Matching results:")
    print(f"    Raw matches  → {len(raw_matches)}")
    print(f"    Good matches → {len(good_matches)} (after Lowe's ratio test)")

    return good_matches


# ─────────────────────────────────────────────
#  STEP 5 — Extract matched point coordinates
# ─────────────────────────────────────────────

def extract_matched_points(kp_left, kp_right, good_matches):
    """
    Extract (x, y) pixel coordinates of matched keypoints.

    Returns:
        pts_left  : (N, 2) array of points in left image
        pts_right : (N, 2) array of corresponding points in right image
    """
    pts_left  = np.float32([kp_left[m.queryIdx].pt  for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

    return pts_left, pts_right


# ─────────────────────────────────────────────
#  STEP 6 — Visualize and save matches
# ─────────────────────────────────────────────

def visualize_matches(img_left, img_right, kp_left, kp_right, good_matches, output_dir):
    """Draw and save the matched keypoints between left and right images."""
    os.makedirs(output_dir, exist_ok=True)

    match_img = cv2.drawMatches(
        img_left,  kp_left,
        img_right, kp_right,
        good_matches[:50],   # Show max 50 matches for clarity
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Convert BGR → RGB for matplotlib
    match_img_rgb = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 6))
    plt.imshow(match_img_rgb)
    plt.title(f"SIFT Matches — {len(good_matches)} good matches found", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "sift_matches.png")
    plt.savefig(output_path, dpi=150)
    plt.show()

    print(f"[✓] Match visualization saved → {output_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_feature_matching(left_path=LEFT_IMAGE_PATH, right_path=RIGHT_IMAGE_PATH):
    """
    Full feature matching pipeline.
    Returns matched point coordinates ready for triangulation.
    """
    print("\n" + "═" * 50)
    print("  SIFT Feature Detection & Matching")
    print("═" * 50)

    # Load
    img_left, img_right, gray_left, gray_right = load_images(left_path, right_path)

    # Detect + Describe
    kp_left, desc_left, kp_right, desc_right = detect_and_compute(gray_left, gray_right)

    # Match
    good_matches = match_descriptors(desc_left, desc_right)

    if len(good_matches) < 8:
        raise ValueError(
            f"Not enough good matches ({len(good_matches)}).\n"
            "Try lowering the LOWE_RATIO or check your images."
        )

    # Extract coordinates
    pts_left, pts_right = extract_matched_points(kp_left, kp_right, good_matches)

    # Visualize
    visualize_matches(img_left, img_right, kp_left, kp_right, good_matches, OUTPUT_DIR)

    print(f"\n[✓] Done. {len(good_matches)} matched point pairs ready for triangulation.\n")

    return pts_left, pts_right


if __name__ == "__main__":
    pts_left, pts_right = run_feature_matching()