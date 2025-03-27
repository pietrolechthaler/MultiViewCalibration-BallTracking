import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# 1. CARICA E PREPARA I DATI
# ---------------------------

# Carica i punti 2D dalle telecamere
with open('annotation-undist/out1-undist.json') as f:
    points_cam1 = json.load(f)
with open('annotation-undist/out4-undist.json') as f:
    points_cam2 = json.load(f)

# Coordinate 3D REALI dei punti del campo (in metri)
points3D_campo = np.array([
    [9, 0, 0],         # C1
    [0, 0, 0],      # C2
    [4.5, 0, 0],    # C3
    [0, 9, 0],      # C4
    [13.5, 0, 0],   # C5
    [18, 0, 0],     # C6
    [18, 9, 0],     # C7
    [13.5, 9, 0],   # C8
    [9, 9, 0],      # C9 
    [4.5, 9, 0]     # C10
], dtype=np.float32)

# Funzione per estrarre punti validi dai JSON
def prepare_points(json_data, points3D_ref):
    points2D = []
    points3D = []
    id_to_idx = {
        '1': 0, '2': 1, '3': 2, '4': 3, '5': 4,
        '6': 5, '7': 6, '8': 7, '9': 8, '10': 9
    }
    for point_id, data in json_data.items():
        if data.get('status') == 'ok':
            idx = id_to_idx[point_id]
            points2D.append([data['coordinates']['x'], data['coordinates']['y']])
            points3D.append(points3D_ref[idx])
    return np.array(points2D, dtype=np.float32), np.array(points3D, dtype=np.float32)

# Prepara i punti per ogni telecamera
points2D_cam1, points3D_cam1 = prepare_points(points_cam1, points3D_campo)
points2D_cam2, points3D_cam2 = prepare_points(points_cam2, points3D_campo)

# ---------------------------
# 2. CALIBRAZIONE ESTRINSECA
# ---------------------------

# Carica le matrici intrinseche 
camera_matrix1 = np.loadtxt('src-gen/out1F-gen/camera_matrix.txt')
dist_coeffs1 = np.loadtxt('src-gen/out1F-gen/distortion_coefficients.txt')
camera_matrix2 = np.loadtxt('src-gen/out4F-gen/camera_matrix.txt')
dist_coeffs2 = np.loadtxt('src-gen/out4F-gen/distortion_coefficients.txt')

# Calibrazione per Camera 1
ret1, rvec1, tvec1 = cv2.solvePnP(
    objectPoints=points3D_cam1,
    imagePoints=points2D_cam1,
    cameraMatrix=camera_matrix1,
    distCoeffs=dist_coeffs1
)

# Calibrazione per Camera 2
ret2, rvec2, tvec2 = cv2.solvePnP(
    objectPoints=points3D_cam2,
    imagePoints=points2D_cam2,
    cameraMatrix=camera_matrix2,
    distCoeffs=dist_coeffs2
)

# Converti i vettori di rotazione in matrici
R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)

# ---------------------------
# 3. MATRICI DI PROIEZIONE
# ---------------------------
proj_matrix1 = camera_matrix1 @ np.hstack((R1, tvec1))
proj_matrix2 = camera_matrix2 @ np.hstack((R2, tvec2))

# ---------------------------
# 4. TRIANGOLAZIONE (ESEMPIO)
# ---------------------------
# Supponiamo di avere un giocatore osservato in:
# Camera 1: [2500, 1200]
# Camera 2: [2600, 1250]
# point2D_cam1 = np.array([2500, 1200], dtype=np.float32)
# point2D_cam2 = np.array([2600, 1250], dtype=np.float32)

# point4D = cv2.triangulatePoints(
#     projMatr1=proj_matrix1,
#     projMatr2=proj_matrix2,
#     projPoints1=point2D_cam1.reshape(-1, 1),
#     projPoints2=point2D_cam2.reshape(-1, 1)
# )
# point3D = (point4D[:3] / point4D[3]).flatten()

# print(f"Posizione 3D del giocatore: X={point3D[0]:.2f}m, Y={point3D[1]:.2f}m, Z={point3D[2]:.2f}m")

# ---------------------------
# 5. VISUALIZZAZIONE
# ---------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Disegna il campo
ax.scatter(points3D_campo[:, 0], points3D_campo[:, 1], points3D_campo[:, 2], 
           c='blue', marker='o', s=50, label='Punti campo')

# # Disegna il giocatore triangolato
# ax.scatter(point3D[0], point3D[1], point3D[2], 
#            c='red', marker='*', s=100, label='Giocatore')

# Disegna le telecamere
ax.quiver(tvec1[0], tvec1[1], tvec1[2], R1[0,0], R1[1,0], R1[2,0], 
          length=1, color='green', label='Camera 1')
ax.quiver(tvec2[0], tvec2[1], tvec2[2], R2[0,0], R2[1,0], R2[2,0], 
          length=1, color='magenta', label='Camera 2')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ricostruzione 3D del campo di pallavolo')
ax.legend()
plt.tight_layout()
plt.show()