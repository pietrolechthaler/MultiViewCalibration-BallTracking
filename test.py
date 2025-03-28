import cv2
import numpy as np
import json

def loadWorldAndImagePoints(cam_id):
    world_points = []
    image_points = []

    WORLD = {
        '1': [0, 9, 0],
        '2': [0, 0, 0],
        '3': [6, 0, 0],
        '4': [9, 0, 0],
        '5': [12, 0, 0],
        '6': [18, 0, 0],
        '7': [18, 9, 0],
        '8': [12, 9, 0],
        '9': [9, 9, 0],
        '10': [6, 9, 0],
        '11': [0, 6.9, 0],
        '12': [0, 2.1, 0],
        '13': [18, 2.1, 0],
        '14': [18, 6.9, 0]
    }

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

        # Converti le liste in array NumPy con dtype float32
        w_points = np.array(world_points, dtype=np.float32)
        i_points = np.array(image_points, dtype=np.float32)

    return w_points, i_points


def main():
    CAMERA_IDS = [1, 2, 3, 4, 6, 7, 8, 12, 13]

    for cam_id in CAMERA_IDS:
        camera_matrix = np.loadtxt(f'./src-gen/out{cam_id}F-gen/camera_matrix.txt', dtype=np.float32)
        dist_coeffs = np.loadtxt(f'./src-gen/out{cam_id}F-gen/distortion_coefficients.txt', dtype=np.float32)

        world_points, image_points = loadWorldAndImagePoints(cam_id)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            world_points, 
            image_points, 
            camera_matrix, 
            dist_coeffs
        )

        # Converti il vettore di rotazione in matrice
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        camera_position = -np.dot(rotation_matrix.T, translation_vector)
        print(f"Posizione della camera {cam_id} nel mondo 3D (X, Y, Z in metri):")
        print(camera_position.flatten())


if __name__ == "__main__":
    main()