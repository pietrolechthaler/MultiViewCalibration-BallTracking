import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils.parameters as parameters
import sys, os
import pickle

def loadWorldAndImagePoints(cam_id):
    world_points = []
    image_points = []

    WORLD = parameters.WORLD_LABEL_POINTS
    label = json.load(open(f'./annotation-dist/out{cam_id}-ann.json'))
    
    # Iterate over the sorted ids
    for key in sorted(label.keys(), key=int):
        
        # Check if the entry is valid
        entry = label[key]
        if entry.get("status") == "ok" and "coordinates" in entry:

            # Extract the 2D coordinates
            coords_2d = entry["coordinates"]
            image_points.append((coords_2d["x"], coords_2d["y"]))
            
            # Extract the 3D coordinates
            if key in WORLD:
                coords_3d = WORLD[key]
                world_points.append(coords_3d)

    w_points = np.array(world_points, dtype=np.float32)
    i_points = np.array(image_points, dtype=np.float32)

    return w_points, i_points


def main():

    CAMERA_IDS = parameters.CAMERA_IDS
    camera_positions = []
    
    # Check if the src-gen folder exists
    SRC_GEN = parameters.SRC_GEN
    if not os.path.exists(SRC_GEN):
        print(f"> Error: The folder {SRC_GEN} does not exist.")
        return
    
    for cam_id in CAMERA_IDS:
        
        # Load the camera matrix and distortion coefficients
        camera_matrix = np.loadtxt(f'./src-gen/out{cam_id}F-gen/camera_matrix.txt', dtype=np.float32)
        dist_coeffs = np.loadtxt(f'./src-gen/out{cam_id}F-gen/distortion_coefficients.txt', dtype=np.float32)

        # Caricare il file .pkl
        with open(f'./src-gen/out{cam_id}F-gen/calibration_data.pkl', 'rb') as file:
            dati_pkl = pickle.load(file)

        # Stampa il valore desiderato dal file .pkl
        valore_da_stampare = dati_pkl['reprojection_error']
        print(f"Valore estratto dal file pkl: {valore_da_stampare}")


        # Get the world and image points for the camera
        world_points, image_points = loadWorldAndImagePoints(cam_id)

        # Solve the PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            world_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs
        )

        # Convert the rotation vector to a rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Compute the camera position
        camera_position = -np.dot(rotation_matrix.T, translation_vector)

        # Invert Z to have positive = up
        # camera_position[2] *= -1
        # rotation_matrix[:, 2] *= -1
        # translation_vector[2] *= -1
        camera_positions.append((cam_id, camera_position, rotation_matrix.T, translation_vector))

        # Print the results
        #print(f"Posizione della camera {cam_id} nel mondo 3D (X, Y, Z in metri):\n")
        #print(f"{camera_position.flatten()}\n")
        #print(f"Matrice di rotazione per camera {cam_id}:\n{rotation_matrix}\n")
        #print(f"Vettore di traslazione per camera {cam_id}:\n{translation_vector.flatten()}\n\n")

        # Save the extrinsic calibration data in a pickle file
        extrinsic_data = {
            'camera_id': cam_id,
            'camera_position': camera_position,
            'rotation_matrix': rotation_matrix,
            'translation_vector': translation_vector
        }
        
        with open(os.path.join(f'./src-gen/out{cam_id}F-gen', 'calibration_extrinsic.pkl'), 'wb') as pkl_file:
            pickle.dump(extrinsic_data, pkl_file)


    # 3D Visualization
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink']
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the field with labels
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

    for i, pt in enumerate(points3D_campo):
        ax.scatter(pt[0], pt[1], pt[2], c='blue', marker='o', s=50)
        ax.text(pt[0], pt[1], pt[2], f'C{i+1}', color='blue')

    # Draw the cameras with labels and axes
    def draw_camera(ax, tvec, R, color, label):
        tvec = tvec.flatten()
        ax.scatter(tvec[0], tvec[1], tvec[2], c=color, s=100, label=label)
        axis_length = 1.5 

        # Camera axes
        for col, axis in zip(['r', 'g', 'b'], [R[:, 0], R[:, 1], R[:, 2]]):
            end_point = tvec + axis_length * axis 
            ax.quiver(tvec[0], tvec[1], tvec[2], axis[0], axis[1], axis[2],
                    length=axis_length, color=col, linewidth=2)

        ax.text(tvec[0], tvec[1], tvec[2], label, color=color)  # Label camera

    # Draw all the cameras
    for idx, (cam_id, camera_position, rotation_matrix, translation_vector) in enumerate(camera_positions):
        color = colors[idx % len(colors)]
        draw_camera(ax, camera_position, rotation_matrix, color, f'Camera {cam_id}')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Visualization of Camera Positions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Save the 3D visualization image
    fig.savefig('./src-gen/camera_positions_3D.png', dpi=300)

if __name__ == "__main__":
    main()
