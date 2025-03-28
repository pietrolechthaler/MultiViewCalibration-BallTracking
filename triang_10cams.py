import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from collections import defaultdict

# ---------------------------
# 1. CONFIGURAZIONE
# ---------------------------

# Lista delle telecamere da utilizzare (1-8, 12-13)
CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
NUM_CAMERAS = len(CAMERA_IDS)
BASE_PATH = 'annotation-undist'
SRC_GEN_PATH = 'src-gen'

# Coordinate 3D REALI dei punti del campo (in metri)
points3D_campo = np.array([
    [0, 9, 0],
    [0, 0, 0],
    [6, 0, 0],
    [9, 0, 0],
    [12, 0, 0],
    [18, 0, 0],
    [18, 9, 0],
    [12, 9, 0],
    [9, 9, 0],
    [6, 9, 0],
    [0, 6.9, 0],
    [0, 2.1, 0],
    [18, 2.1, 0],
    [18, 6.9, 0]
], dtype=np.float32)

# ---------------------------
# 2. CARICAMENTO DATI
# ---------------------------

def load_camera_data(cam_id):
    """Carica i dati per una singola telecamera"""
    # Carica punti 2D
    json_path = f'{BASE_PATH}/out{cam_id}-undist.json'
    with open(json_path) as f:
        points_json = json.load(f)
    
    # Carica parametri intrinseci
    cam_folder = f'out{cam_id}F-gen' 
    cam_matrix_path = f'{SRC_GEN_PATH}/{cam_folder}/camera_matrix.txt'
    dist_coeffs_path = f'{SRC_GEN_PATH}/{cam_folder}/distortion_coefficients.txt'
    
    points2D, points3D = prepare_points(points_json, points3D_campo)
    camera_matrix = np.loadtxt(cam_matrix_path)
    dist_coeffs = np.loadtxt(dist_coeffs_path)
    
    return {
        'points2D': points2D,
        'points3D': points3D,
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs,
        'id': cam_id
    }

def prepare_points(json_data, points3D_ref):
    """Estrae punti validi dai JSON"""
    points2D = []
    points3D = []
    id_to_idx = {str(i): i-1 for i in range(1, 11)}  # Mappa '1'->0, ..., '10'->9
    for point_id, data in json_data.items():
        if data.get('status') == 'ok':
            idx = id_to_idx[point_id]
            points2D.append([data['coordinates']['x'], data['coordinates']['y']])
            points3D.append(points3D_ref[idx])
    return np.array(points2D, dtype=np.float32), np.array(points3D, dtype=np.float32)

# Carica dati per tutte le telecamere
camera_data = {cam_id: load_camera_data(cam_id) for cam_id in CAMERA_IDS}

# ---------------------------
# 3. CALIBRAZIONE ESTRINSECA
# ---------------------------

for cam_id, data in camera_data.items():
    ret, rvec, tvec = cv2.solvePnP(
        objectPoints=data['points3D'],
        imagePoints=data['points2D'],
        cameraMatrix=data['camera_matrix'],
        distCoeffs=data['dist_coeffs']
    )
    R, _ = cv2.Rodrigues(rvec)
    data['R'] = R
    data['tvec'] = tvec
    data['proj_matrix'] = data['camera_matrix'] @ np.hstack((R, tvec))

# ---------------------------
# 4. TRIANGOLAZIONE A COPPIE + FUSIONE
# ---------------------------

def triangulate_all_pairs(camera_data, point2D_dict):
    """
    Triangola un punto usando tutte le coppie possibili di telecamere
    :param point2D_dict: Dizionario {cam_id: [x,y]} con i punti osservati
    :return: Punto 3D mediato da tutte le coppie valide
    """
    valid_cams = [cam_id for cam_id in point2D_dict if cam_id in camera_data]
    if len(valid_cams) < 2:
        return None
    
    all_points = []
    
    # Genera tutte le coppie possibili
    from itertools import combinations
    for (cam1, cam2) in combinations(valid_cams, 2):
        try:
            P1 = camera_data[cam1]['proj_matrix']
            P2 = camera_data[cam2]['proj_matrix']
            point1 = np.array(point2D_dict[cam1], dtype=np.float32)
            point2 = np.array(point2D_dict[cam2], dtype=np.float32)
            
            point4D = cv2.triangulatePoints(P1, P2, point1.reshape(-1,1), point2.reshape(-1,1))
            point3D = (point4D[:3]/point4D[3]).flatten()
            all_points.append(point3D)
        except:
            continue
    
    if not all_points:
        return None
    
    # Calcola la media e la deviazione standard
    all_points = np.array(all_points)
    mean_point = np.mean(all_points, axis=0)
    std_dev = np.std(all_points, axis=0)
    
    # Filtra outliers (più di 2 deviazioni standard)
    filtered_points = []
    for pt in all_points:
        if np.all(np.abs(pt - mean_point) < 2*std_dev):
            filtered_points.append(pt)
    
    return np.mean(filtered_points, axis=0) if filtered_points else mean_point

# Esempio di utilizzo:
# point2D_dict = {
#     1: [2500, 1200],  # Camera 1
#     4: [2600, 1250],  # Camera 4
#     12: [2550, 1300]  # Camera 12
# }
# point3D = triangulate_all_pairs(camera_data, point2D_dict)
# print(f"Punto 3D triangolato: {point3D}")

# ---------------------------
# 5. VISUALIZZAZIONE
# ---------------------------

def plot_camera_setup(camera_data, points3D_campo):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Disegna il campo
    ax.scatter(points3D_campo[:, 0], points3D_campo[:, 1], points3D_campo[:, 2],
               c='blue', marker='o', s=50, label='Punti campo')
    
    # Disegna tutte le telecamere
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CAMERAS))
    for i, (cam_id, data) in enumerate(camera_data.items()):
        tvec = data['tvec']
        R = data['R']
        ax.quiver(tvec[0], tvec[1], tvec[2], R[0,0], R[1,0], R[2,0],
                  length=1, color=colors[i], label=f'Camera {cam_id}')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Setup {NUM_CAMERAS} telecamere (IDs: {CAMERA_IDS})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_camera_setup(camera_data, points3D_campo)