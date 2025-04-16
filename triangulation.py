import cv2
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TRAIN_FOLDER = './tracking/runs/detect/train4'

CAM_PAIRS = [
        ["3", "1"],
        ["1", "2"],
        ["2", "4"],
        ["4", "12"],
        ["12", "7"],
        ["7", "8"],
        ["8", "6"],
        ["6", "5"],
        ["5", "13"],
        ["13", "3"]
    ]

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

def plot_court(coords_3d_df, cam_id1, cam_id2):
    # Converti in array
    coords_3d_np = coords_3d_df[['X', 'Y', 'Z']].to_numpy()

    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    draw_volleyball_court(ax)
    ax.scatter(coords_3d_np[:, 0], coords_3d_np[:, 1], coords_3d_np[:, 2],
            c='yellow', marker='o', label='Ball Detection')

    ax.set_xlabel('X (court length) [m]')
    ax.set_ylabel('Y (court width) [m]')
    ax.set_zlabel('Z (height) [m]')

    ax.set_xlim(-COURT_LENGTH/2 - 1, COURT_LENGTH/2 + 1)
    ax.set_ylim(-COURT_WIDTH/2 - 1, COURT_WIDTH/2 + 1)
    ax.set_zlim(0, 5)

    ax.set_title(f'3D Ball Detection over Volleyball Court - Cameras {cam_id1} and {cam_id2}')
    ax.legend()
    plt.tight_layout()
    plt.show()

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


for cam_pair in CAM_PAIRS:

    cam_id1 = int(cam_pair[0])
    cam_id2 = int(cam_pair[1])

    # Caricamento dei dati
    df_cam1 = pd.read_csv(os.path.join(TRAIN_FOLDER, f'coordinates/out{cam_id1}_coordinates.csv'))
    df_cam2 = pd.read_csv(os.path.join(TRAIN_FOLDER, f'coordinates/out{cam_id2}_coordinates.csv'))

    # Rimuovi spazi extra dalle intestazioni
    df_cam1.columns = df_cam1.columns.str.strip()
    df_cam2.columns = df_cam2.columns.str.strip()

    # Rinominare le colonne per chiarezza dopo join
    df_cam1 = df_cam1.rename(columns={'x_center': f'x_cam{cam_id1}', 'y_center': f'y_cam{cam_id1}'})
    df_cam2 = df_cam2.rename(columns={'x_center': f'x_cam{cam_id2}', 'y_center': f'y_cam{cam_id2}'})

    # Unione sui sync_idx
    merged = pd.merge(df_cam1, df_cam2, on='timestamp_sec', suffixes=(f'_cam{cam_id1}', f'_cam{cam_id2}'))

    # Caricamento matrici di proiezione
    P1 = load_ProjectionMatrix(cam_id1)
    P2 = load_ProjectionMatrix(cam_id2)

    coords_3d = []

    # Estrai le coordinate 2D da entrambe le telecamere
    points2d_cam1 = merged[[f'x_cam{cam_id1}', f'y_cam{cam_id1}']].to_numpy().T  # Shape (2, N)
    points2d_cam2 = merged[[f'x_cam{cam_id2}', f'y_cam{cam_id2}']].to_numpy().T  # Shape (2, N)

    # Triangolazione di tutti i punti
    points4d = cv2.triangulatePoints(P1, P2, points2d_cam1, points2d_cam2)

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
    #print(coords_3d_df)

    # Salva il DataFrame in un file CSV
    coords_3d_df.to_csv(os.path.join(TRAIN_FOLDER, f'coordinates/out{cam_id1}_{cam_id2}_coordinates.csv'), index=False)

    plot_court(coords_3d_df, cam_id1, cam_id2)

