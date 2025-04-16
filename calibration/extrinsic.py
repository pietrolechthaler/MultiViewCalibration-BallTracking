import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils.parameters as parameters
import os
import pickle

def loadWorldAndImagePoints(cam_id):
    world_points = []
    image_points = []

    WORLD = parameters.WORLD_LABEL_POINTS
    label = json.load(open(f'./annotation/annotation-dist/out{cam_id}-ann.json'))
    
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

def visualize_camera_positions(camera_positions, real_camera_positions=None):
    # 3D Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the field with labels
    points3D_campo = np.array(list(parameters.WORLD_LABEL_POINTS.values()), dtype=np.float32)

    for i, pt in enumerate(points3D_campo):
        ax.scatter(pt[0], pt[1], pt[2], c='red', marker='o', s=30)
        ax.text(pt[0], pt[1], pt[2], f'C{i+1}', color='red', fontsize=8)

    # Draw the cameras with labels and axes
    def draw_camera(ax, tvec, R, color, label):
        tvec = tvec.flatten()
        ax.scatter(tvec[0], tvec[1], tvec[2], c=color, s=100, label=label)
        axis_length = 3

        # Only draw the Z-axis
        z_axis = R[:, 2]  # Get the Z-axis direction
        end_point = tvec + axis_length * z_axis 
        ax.quiver(tvec[0], tvec[1], tvec[2], z_axis[0], z_axis[1], z_axis[2],
                length=axis_length, color='b', linewidth=2)  # Draw Z-axis in blue

        ax.text(tvec[0], tvec[1], tvec[2], label, color=color, fontsize=10)  # Label camera

    # Draw all the cameras
    for idx, (cam_id, camera_position, rotation_matrix, translation_vector) in enumerate(camera_positions):
        color = 'blue'  # Set a fixed color for all cameras
        draw_camera(ax, camera_position, rotation_matrix, color, f'Calc {cam_id}')

    # Draw real camera positions if provided
    if real_camera_positions is not None:
        for idx, real_position in enumerate(real_camera_positions):
            ax.scatter(real_position[0], real_position[1], real_position[2], c='orange', s=100, label=f'Real Camera {idx+1}')
            ax.text(real_position[0], real_position[1], real_position[2], f'Real {idx+1}', color='orange')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Visualization of Camera Positions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Save the 3D visualization image
    fig.savefig('./src-gen/camera_positions_3D.png', dpi=300)


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
        camera_positions.append((cam_id, camera_position, rotation_matrix.T, translation_vector))

        # Print the results
        print(f"Camera {cam_id} :" , camera_position.flatten())

        # Save the extrinsic calibration data in a pickle file
        extrinsic_data = {
            'camera_id': cam_id,
            'camera_position': camera_position,
            'rotation_matrix': rotation_matrix.T,
            'translation_vector': translation_vector,
        }
        
        with open(os.path.join(f'./src-gen/out{cam_id}F-gen', 'calibration_extrinsic.pkl'), 'wb') as pkl_file:
            pickle.dump(extrinsic_data, pkl_file)

    # Call the visualization function
    visualize_camera_positions(camera_positions, parameters.REAL_WORLD_CAMERA_POSITIONS)

if __name__ == "__main__":
    main()
