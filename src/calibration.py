import numpy as np
import cv2 as cv
import glob
import os

output_folder = '../output/corners_detection'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


square_size = 25  # Taille du carré en mm
pattern_size = (8, 6)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparation des points objets (0,0,0), (25,0,0), (50,0,0) ..., (175,125,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('./data/calibration/*.jpg')
print(len(images))
# Avant la boucle for
cv.namedWindow('img', cv.WINDOW_NORMAL)
cv.resizeWindow('img', 1000, 800) 
image_size = None

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Impossible de lire : {fname}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]
        # print(f'image size : {image_size}')


    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        # --- SAVE at corner_detection ---
        base_name = os.path.basename(fname) 
        save_path = os.path.join("./output/corners_detection", "drawchessboard_" + base_name)
        
        cv.imwrite(save_path, img) # Sauvegarde l'image avec les dessins
        # ----------------------------------
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        # Si la détection échoue, on affiche le nom du fichier pour débugger
        print(f"Échec de détection pour : {fname}")

cv.destroyAllWindows()

# Clibration

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)


if ret:
    print("Calibration réussie !")
    print("Matrice Intrinsèque (K) :\n", mtx)
    print("Distorsion : \n",dist)
    # 4. Sauvegarde des paramètres pour la suite du projet
    cv_file = cv.FileStorage("../calibration_results/camera_params.xml", cv.FileStorage_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    cv_file.release()
    print("Paramètres sauvegardés dans camera_params.xml")