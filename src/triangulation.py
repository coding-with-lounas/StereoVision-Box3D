"""
triangulation.py
----------------
Calcul des coordonnées 3D à partir des paires de points SIFT + calibration + baseline.

Input  : pts_left, pts_right (depuis features.py) + camera_params.xml + BASELINE
Output : nuage_boites.ply
"""

import cv2 as cv
import numpy as np
import open3d as o3d
import os

# ─────────────────────────────────────────────
#  CONFIG  ← à modifier selon ton acquisition
# ─────────────────────────────────────────────

CALIBRATION_FILE = './calibration_results/camera_params.xml'
OUTPUT_DIR       = './output/reconstruction/'
OUTPUT_PLY       = os.path.join(OUTPUT_DIR, 'nuage_boites.ply')

BASELINE = 100.0   # ← Distance entre les 2 positions de ta caméra EN MM
                   #   (même unité que square_size dans calibration.py)

# ─────────────────────────────────────────────
#  ÉTAPE 1 — Charger les paramètres de calibration
# ─────────────────────────────────────────────

def load_calibration(path=CALIBRATION_FILE):
    """Lit la matrice K et les coefficients de distorsion depuis le fichier XML."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fichier de calibration introuvable : {path}\n"
            "Lance d'abord calibration.py !"
        )

    fs = cv.FileStorage(path, cv.FILE_STORAGE_READ)
    K    = fs.getNode("K").mat()
    dist = fs.getNode("D").mat()
    fs.release()

    print(f"[✓] Calibration chargée depuis : {path}")
    print(f"    Focale fx = {K[0,0]:.2f} px,  fy = {K[1,1]:.2f} px")
    print(f"    Centre optique : cx = {K[0,2]:.2f},  cy = {K[1,2]:.2f}")
    return K, dist


# ─────────────────────────────────────────────
#  ÉTAPE 2 — Corriger la distorsion des points
# ─────────────────────────────────────────────

def undistort_points(pts_left, pts_right, K, dist):
    """
    Corrige la distorsion de l'objectif sur les coordonnées 2D des points appariés.
    Sans cette étape, le calcul 3D sera biaisé.
    """
    pts_left_ud  = cv.undistortPoints(
        pts_left.reshape(-1, 1, 2),  K, dist, P=K
    ).reshape(-1, 2)

    pts_right_ud = cv.undistortPoints(
        pts_right.reshape(-1, 1, 2), K, dist, P=K
    ).reshape(-1, 2)

    print(f"[✓] Distorsion corrigée sur {len(pts_left_ud)} paires de points.")
    return pts_left_ud, pts_right_ud


# ─────────────────────────────────────────────
#  ÉTAPE 3 — Construire les matrices de projection
# ─────────────────────────────────────────────

def build_projection_matrices(K, baseline):
    """
    Construit P_gauche et P_droite.

    La caméra gauche est à l'origine :
        P_gauche = K × [I | 0]

    La caméra droite est décalée de 'baseline' sur l'axe X :
        P_droite = K × [I | -b, 0, 0]

    Formule de la disparité :
        Z = (f × b) / (u_gauche - u_droite)
    """
    # Caméra gauche — à l'origine
    P_gauche = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Caméra droite — décalée de baseline sur X
    t_droite = np.array([[-baseline], [0.0], [0.0]])
    P_droite = K @ np.hstack((np.eye(3), t_droite))

    print(f"[✓] Matrices de projection construites (baseline = {baseline} mm)")
    return P_gauche, P_droite


# ─────────────────────────────────────────────
#  ÉTAPE 4 — Triangulation → coordonnées 3D
# ─────────────────────────────────────────────

def triangulate(pts_left, pts_right, P_gauche, P_droite):
    """
    Calcule les coordonnées 3D (X, Y, Z) pour chaque paire de points appariés.

    cv.triangulatePoints résout le système :
        u_gauche  = P_gauche × X
        u_droite  = P_droite × X

    Retourne un tableau (N, 3) de points 3D.
    """
    # triangulatePoints attend des tableaux (2, N)
    points_4d = cv.triangulatePoints(
        P_gauche, P_droite,
        pts_left.T,
        pts_right.T
    )

    # Conversion coordonnées homogènes → 3D : diviser par W
    points_3d = (points_4d[:3] / points_4d[3]).T   # shape (N, 3)

    # Filtrer les points derrière la caméra (Z négatif ou nul)
    masque    = points_3d[:, 2] > 0
    points_3d = points_3d[masque]

    print(f"[✓] Triangulation terminée :")
    print(f"    Points totaux    : {len(masque)}")
    print(f"    Points valides   : {masque.sum()}  (Z > 0)")
    print(f"    Profondeur Z     : {points_3d[:, 2].min():.1f} mm"
          f" → {points_3d[:, 2].max():.1f} mm")

    return points_3d


# ─────────────────────────────────────────────
#  ÉTAPE 5 — Sauvegarder le nuage de points
# ─────────────────────────────────────────────

def save_point_cloud(points_3d, output_path=OUTPUT_PLY):
    """Sauvegarde le nuage de points au format .ply (MeshLab, CloudCompare, Open3D)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Colorier par profondeur (Z) : bleu = proche, rouge = loin
    z      = points_3d[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm, np.zeros_like(z_norm), 1 - z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_path, pcd)
    print(f"[✓] Nuage de points sauvegardé → {output_path}")


# ─────────────────────────────────────────────
#  ÉTAPE 6 — Visualiser le nuage de points
# ─────────────────────────────────────────────

def visualize_point_cloud(output_path=OUTPUT_PLY):
    """Ouvre une fenêtre 3D interactive pour visualiser le nuage."""
    pcd = o3d.io.read_point_cloud(output_path)
    print("[✓] Visualisation 3D lancée (ferme la fenêtre pour continuer)...")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="StereoVision — Nuage de points 3D",
        width=1024,
        height=768
    )


# ─────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def run_triangulation(pts_left, pts_right, baseline=BASELINE):
    """
    Pipeline complet de triangulation.

    Args:
        pts_left  : (N, 2) coordonnées des points dans l'image gauche
        pts_right : (N, 2) coordonnées des points correspondants dans l'image droite
        baseline  : distance entre les deux positions de la caméra (en mm)

    Returns:
        points_3d : (N, 3) coordonnées 3D reconstruites
    """
    print("\n" + "═" * 50)
    print("  Triangulation — Calcul des coordonnées 3D")
    print("═" * 50)

    K, dist                    = load_calibration()
    pts_left_ud, pts_right_ud  = undistort_points(pts_left, pts_right, K, dist)
    P_gauche, P_droite         = build_projection_matrices(K, baseline)
    points_3d                  = triangulate(pts_left_ud, pts_right_ud, P_gauche, P_droite)

    save_point_cloud(points_3d)
    visualize_point_cloud()

    print(f"\n[✓] Reconstruction terminée — {len(points_3d)} points 3D reconstruits.\n")
    return points_3d


# ─────────────────────────────────────────────
#  EXÉCUTION STANDALONE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from features import run_feature_matching
    pts_left, pts_right = run_feature_matching()
    run_triangulation(pts_left, pts_right)