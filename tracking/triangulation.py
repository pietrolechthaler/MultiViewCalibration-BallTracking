import cv2
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parameters import TRACKING_FOLDER, CAM_PAIRS, COURT_LENGTH, COURT_WIDTH, NET_HEIGHT, NET_WIDTH, START_SEC, END_SEC



def draw_volleyball_court(ax):
    """Draw a 3D volleyball court on the given matplotlib axis"""
    hl = COURT_LENGTH / 2  # half length
    hw = COURT_WIDTH / 2   # half width

    # Perimeter lines
    ax.plot([-hl, hl], [-hw, -hw], [0, 0], 'white')
    ax.plot([-hl, hl], [hw, hw], [0, 0], 'white')
    ax.plot([-hl, -hl], [-hw, hw], [0, 0], 'white')
    ax.plot([hl, hl], [-hw, hw], [0, 0], 'white')

    # Center line (dashed)
    ax.plot([-hl, hl], [0, 0], [0, 0], 'white', linestyle='--')

    # Attack lines (3 meters from center)
    attack_line = 3.0  
    ax.plot([attack_line, attack_line], [-hw, hw], [0, 0], 'white')
    ax.plot([-attack_line, -attack_line], [-hw, hw], [0, 0], 'white')

    # Net (red)
    ax.plot([-NET_WIDTH/2, NET_WIDTH/2], [0, 0], [NET_HEIGHT, NET_HEIGHT], 'red')
    ax.plot([0, 0], [-hw, hw], [NET_HEIGHT, NET_HEIGHT], 'red', linestyle=':')

    # Net poles
    ax.plot([0, 0], [-hw, -hw], [0, NET_HEIGHT], 'red')
    ax.plot([0, 0], [hw, hw], [0, NET_HEIGHT], 'red')

def plot_court(coords_3d_df, cam_id1, cam_id2):
    """Plot 3D ball coordinates over a volleyball court for a camera pair"""
    # Convert to numpy array
    coords_3d_np = coords_3d_df[['X', 'Y', 'Z']].to_numpy()

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw court and plot ball detections
    draw_volleyball_court(ax)
    ax.scatter(coords_3d_np[:, 0], coords_3d_np[:, 1], coords_3d_np[:, 2],
            c='yellow', marker='o', label='Ball Detection')

    # Set axis labels and limits
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
    """Load camera projection matrix from calibration files"""
    folder = f'./src-gen/out{cam_id}F-gen'
    
    # Load intrinsic parameters (camera matrix, distortion coefficients)
    with open(os.path.join(folder, 'calibration_data.pkl'), 'rb') as f:
        intrinsic = pickle.load(f)
    # Load extrinsic parameters (rotation, translation)
    with open(os.path.join(folder, 'calibration_extrinsic.pkl'), 'rb') as f:
        extrinsic = pickle.load(f)

    K = intrinsic['camera_matrix']  # Intrinsic matrix
    R = extrinsic['rotation_matrix']  # Rotation matrix
    C = extrinsic['camera_position']  # Camera position

    # Build extrinsic matrix
    extrinsic_matrix = np.hstack((R, C))
    extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

    # Invert to get world-to-camera transform
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    extrinsic_matrix_3x4 = extrinsic_matrix[:3, :]
    
    # Compute full projection matrix (K * [R|t])
    P = np.dot(K, extrinsic_matrix_3x4)

    return P


# Main processing loop for each camera pair
for cam_pair in CAM_PAIRS:
    cam_id1 = int(cam_pair[0])
    cam_id2 = int(cam_pair[1])

    # Load detection data for both cameras
    df_cam1 = pd.read_csv(os.path.join(TRACKING_FOLDER, f'coordinates/out{cam_id1}_coordinates.csv'))
    df_cam2 = pd.read_csv(os.path.join(TRACKING_FOLDER, f'coordinates/out{cam_id2}_coordinates.csv'))

    # Clean column names (remove extra spaces)
    df_cam1.columns = df_cam1.columns.str.strip()
    df_cam2.columns = df_cam2.columns.str.strip()

    # Rename columns for clarity after join
    df_cam1 = df_cam1.rename(columns={'x_center': f'x_cam{cam_id1}', 'y_center': f'y_cam{cam_id1}'})
    df_cam2 = df_cam2.rename(columns={'x_center': f'x_cam{cam_id2}', 'y_center': f'y_cam{cam_id2}'})

    # Merge on timestamp to find matching detections
    merged = pd.merge(df_cam1, df_cam2, on='timestamp_sec', suffixes=(f'_cam{cam_id1}', f'_cam{cam_id2}'))

    # Load projection matrices for both cameras
    P1 = load_ProjectionMatrix(cam_id1)
    P2 = load_ProjectionMatrix(cam_id2)

    # Extract 2D coordinates from both cameras
    points2d_cam1 = merged[[f'x_cam{cam_id1}', f'y_cam{cam_id1}']].to_numpy().T  # Shape (2, N)
    points2d_cam2 = merged[[f'x_cam{cam_id2}', f'y_cam{cam_id2}']].to_numpy().T  # Shape (2, N)

    # Triangulate 3D points from 2D correspondences
    points4d = cv2.triangulatePoints(P1, P2, points2d_cam1, points2d_cam2)

    # Convert from homogeneous to Cartesian coordinates
    points3d = cv2.convertPointsFromHomogeneous(points4d.T).reshape(-1, 3)

    # Create DataFrame with 3D coordinates and timestamps
    coords_3d_df = pd.DataFrame({
        'timestamp_sec': merged['timestamp_sec'],
        'X': points3d[:, 0],  # Court length axis
        'Y': points3d[:, 1],  # Court width axis
        'Z': points3d[:, 2]   # Height axis
    })

    # Create extended DataFrame with camera pair info
    coords_3d_df_all = pd.DataFrame({
        'cam_id1': cam_id1,
        'cam_id2': cam_id2,
        'timestamp_sec': merged['timestamp_sec'],
        'X': points3d[:, 0],
        'Y': points3d[:, 1],
        'Z': points3d[:, 2]
    })

    # Save per-camera-pair results
    coords_3d_df.to_csv(os.path.join(TRACKING_FOLDER, f'coordinates/out{cam_id1}_{cam_id2}_coordinates.csv'), index=False)

    # Append to combined results file
    if cam_id1 == 3 and cam_id2 == 1:  # First pair - create new file
        coords_3d_df_all.to_csv(os.path.join(TRACKING_FOLDER, f'coordinates/coords_3d_all.csv'), index=False)
    else:  # Subsequent pairs - append to existing file
        coords_3d_df_all.to_csv(os.path.join(TRACKING_FOLDER, f'coordinates/coords_3d_all.csv'), mode='a', header=False, index=False)

    # Plot for this camera pair
    #plot_court(coords_3d_df, cam_id1, cam_id2)

# Sort the final combined CSV file by timestamp
df = pd.read_csv(os.path.join(TRACKING_FOLDER, f'coordinates/coords_3d_all.csv'))
# Remove rows with TIMESTAMP_SEC < START_SEC or > END_SEC
df = df[(df['timestamp_sec'] >= START_SEC) & (df['timestamp_sec'] <= END_SEC)]
df = df.sort_values(by='timestamp_sec')
df.to_csv(os.path.join(TRACKING_FOLDER, f'coordinates/coords_3d_all.csv'), index=False)

# Extract 3D coordinates for final visualization
coords_3d = df[['X', 'Y', 'Z']]

# Create the final plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the court
draw_volleyball_court(ax)

# Plot complete ball trajectory
ax.scatter(coords_3d['X'], coords_3d['Y'], coords_3d['Z'], 
            c='yellow', marker='o', s=20, label='Ball positions')

# Configure axes
ax.set_xlabel('Court length (X) [m]')
ax.set_ylabel('Court width (Y) [m]')
ax.set_zlabel('Height (Z) [m]')

ax.set_xlim(-COURT_LENGTH/2 - 1, COURT_LENGTH/2 + 1)
ax.set_ylim(-COURT_WIDTH/2 - 1, COURT_WIDTH/2 + 1)
ax.set_zlim(0, 5)

ax.set_title('3D Ball Detection over Volleyball Court - All Cameras')
ax.legend()

plt.tight_layout()
plt.show()

# Save the final plot as an image
fig.savefig(os.path.join(TRACKING_FOLDER, 'coordinates/3D_ball_detections.png'), dpi=300)