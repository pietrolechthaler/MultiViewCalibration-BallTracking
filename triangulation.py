import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import calibration.utils.parameters as parameters

# ---------------------------
# 1. CARICA E PREPARA I DATI
# ---------------------------
# Controlla se sono stati forniti i numeri delle telecamere
if len(sys.argv) != 3:
    print("Uso: python train.py <numero_camera_1> <numero_camera_2>")
    sys.exit(1)

# Ottieni i numeri delle telecamere dagli argomenti della riga di comando
cam1_id = sys.argv[1]
cam2_id = sys.argv[2]

# Carica i punti 2D dalle telecamere
with open(f'annotation-undist/out{cam1_id}-undist.json') as f:
    points_cam1 = json.load(f)
with open(f'annotation-undist/out{cam2_id}-undist.json') as f:
    points_cam2 = json.load(f)

# Coordinate 3D REALI dei punti del campo (in metri)
points3D_campo = np.array([
    parameters.WORLD_LABEL_POINTS
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
camera_matrix1 = np.loadtxt(f'src-gen/out{cam1_id}F-gen/camera_matrix.txt')
dist_coeffs1 = np.loadtxt(f'src-gen/out{cam1_id}F-gen/distortion_coefficients.txt')
camera_matrix2 = np.loadtxt(f'src-gen/out{cam2_id}F-gen/camera_matrix.txt')
dist_coeffs2 = np.loadtxt(f'src-gen/out{cam2_id}F-gen/distortion_coefficients.txt')

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
# 6. STAMPA DEI PARAMETRI DELLE TELECAMERE
# ---------------------------

def print_camera_params(cam_id, camera_matrix, tvec, points3D):
    """Stampa i parametri della telecamera in formato leggibile"""
    print(f"\nCamera {cam_id}:")
    print("Matrice di camera:")
    print(np.array2string(camera_matrix, suppress_small=True, precision=6))
    
    # Calcola distanza media ai punti
    distances = [np.linalg.norm(tvec.flatten() - pt) for pt in points3D]
    avg_distance = np.mean(distances)
    
    print("Vettore di traslazione (tvec):")
    print(np.array2string(tvec.flatten(), suppress_small=True, precision=6))
    print(f"Distanza media ai punti: {avg_distance:.2f} unità")



# Stampa i parametri per le telecamere specificate
print_camera_params(cam1_id, camera_matrix1, tvec1, points3D_cam1)
print_camera_params(cam2_id, camera_matrix2, tvec2, points3D_cam2)

# ---------------------------
# 7. VISUALIZZAZIONE MIGLIORATA
# ---------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Disegna il campo con etichette
for i, pt in enumerate(points3D_campo):
    ax.scatter(pt[0], pt[1], pt[2], c='blue', marker='o', s=50)
    ax.text(pt[0], pt[1], pt[2], f'C{i+1}', color='blue')

# Disegna le telecamere con assi di orientamento
def draw_camera(ax, tvec, R, color, label):
    tvec = tvec.flatten()
    ax.scatter(tvec[0], tvec[1], tvec[2], c=color, s=100, label=label)
    
    # Disegna i 3 assi della telecamera
    axis_length = 0.5
    for col, axis in zip(['r','g','b'], [R[:,0], R[:,1], R[:,2]]):
        ax.quiver(tvec[0], tvec[1], tvec[2],
                 axis[0], axis[1], axis[2],
                 length=axis_length, color=col, linewidth=2)

draw_camera(ax, tvec1, R1, 'green', f'Camera {cam1_id}')
draw_camera(ax, tvec2, R2, 'magenta', f'Camera {cam2_id}')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Ricostruzione 3D del campo di pallavolo\n(Visualizzazione assi telecamere)')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#save the figure with name of the cameras
fig.savefig(f'camera_{cam1_id}_{cam2_id}_3D_plot.png', bbox_inches='tight')

