import cv2
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_ProjectionMatrix(cam_id):
    folder = f'./src-gen/out{cam_id}F-gen'
    
    with open(os.path.join(folder, 'calibration_data.pkl'), 'rb') as f:
        intrinsic = pickle.load(f)
    with open(os.path.join(folder, 'calibration_extrinsic.pkl'), 'rb') as f:
        extrinsic = pickle.load(f)

    K = intrinsic['camera_matrix']
    R = extrinsic['rotation_matrix']
    C = extrinsic['camera_position']

    extrinsic_matrix = np.hstack((R, C))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    P = np.dot(K, extrinsic_matrix_3x4)

    return P

# Caricamento dei dati
df_cam1 = pd.read_csv('./tracking/runs/detect/train4/coordinates/out1_coordinates.csv')
df_cam3 = pd.read_csv('./tracking/runs/detect/train4/coordinates/out6_coordinates.csv')

# Rimuovi spazi extra dalle intestazioni
df_cam1.columns = df_cam1.columns.str.strip()
df_cam3.columns = df_cam3.columns.str.strip()

# Rinominare le colonne per chiarezza dopo join
df_cam1 = df_cam1.rename(columns={'x_center': 'x_cam1', 'y_center': 'y_cam1'})
df_cam3 = df_cam3.rename(columns={'x_center': 'x_cam3', 'y_center': 'y_cam3'})

# Unione sui sync_idx
merged = pd.merge(df_cam1, df_cam3, on='timestamp_sec', suffixes=('_cam1', '_cam3'))

# Caricamento matrici di proiezione
P1 = load_ProjectionMatrix(1)
P3 = load_ProjectionMatrix(6)

coords_3d = []

# Estrai le coordinate 2D da entrambe le telecamere
points2d_cam1 = merged[['x_cam1', 'y_cam1']].to_numpy().T  # Shape (2, N)
points2d_cam3 = merged[['x_cam3', 'y_cam3']].to_numpy().T  # Shape (2, N)

# Triangolazione di tutti i punti
points4d = cv2.triangulatePoints(P1, P3, points2d_cam1, points2d_cam3)

# Conversione a coordinate 3D (da omogenee a cartesiane)
points3d = cv2.convertPointsFromHomogeneous(points4d.T).reshape(-1, 3)

# Combina con gli indici di sincronizzazione
coords_3d_df = pd.DataFrame({
    'timestamp_sec': merged['timestamp_sec'],
    'X': points3d[:, 0],
    'Y': points3d[:, 1],
    'Z': points3d[:, 2]
})

# Risultato in DataFrame
print(coords_3d_df)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Costanti del campo
COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_HEIGHT = 2.24
NET_WIDTH = 1.0

def draw_volleyball_court(ax):
    hl = COURT_LENGTH / 2
    hw = COURT_WIDTH / 2

    # Linee perimetrali
    ax.plot([-hl, hl], [-hw, -hw], [0, 0], 'white')
    ax.plot([-hl, hl], [hw, hw], [0, 0], 'white')
    ax.plot([-hl, -hl], [-hw, hw], [0, 0], 'white')
    ax.plot([hl, hl], [-hw, hw], [0, 0], 'white')

    # Linea centrale
    ax.plot([-hl, hl], [0, 0], [0, 0], 'white', linestyle='--')

    # Linee d’attacco
    attack_line = 3.0
    ax.plot([attack_line, attack_line], [-hw, hw], [0, 0], 'white')
    ax.plot([-attack_line, -attack_line], [-hw, hw], [0, 0], 'white')

    # Rete
    ax.plot([-NET_WIDTH/2, NET_WIDTH/2], [0, 0], [NET_HEIGHT, NET_HEIGHT], 'red')
    ax.plot([0, 0], [-hw, hw], [NET_HEIGHT, NET_HEIGHT], 'red', linestyle=':')

    # Pali
    ax.plot([0, 0], [-hw, -hw], [0, NET_HEIGHT], 'red')
    ax.plot([0, 0], [hw, hw], [0, NET_HEIGHT], 'red')

# Converti in array
coords_3d_np = coords_3d_df[['X', 'Y', 'Z']].to_numpy()

# Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

draw_volleyball_court(ax)
ax.scatter(coords_3d_np[:, 0], coords_3d_np[:, 1], coords_3d_np[:, 2],
           c='yellow', marker='o', label='Ball trajectory')

ax.set_xlabel('X (court length) [m]')
ax.set_ylabel('Y (court width) [m]')
ax.set_zlabel('Z (height) [m]')

ax.set_xlim(-COURT_LENGTH/2 - 1, COURT_LENGTH/2 + 1)
ax.set_ylim(-COURT_WIDTH/2 - 1, COURT_WIDTH/2 + 1)
ax.set_zlim(0, 5)

ax.set_title('3D Ball Trajectory over Volleyball Court')
ax.legend()
plt.tight_layout()
plt.show()

