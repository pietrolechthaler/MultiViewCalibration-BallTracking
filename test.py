import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
    camera_positions = []
    
    # Definisci i colori per le telecamere
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'pink']
    
    # Apri un file di testo per scrivere i risultati
    with open('camera_positions.txt', 'w') as f:
        for idx, cam_id in enumerate(CAMERA_IDS):
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

            # Inverti Z per avere positivo=alto (se preferisci)
            camera_position[2] *= -1
            rotation_matrix[:,2] *= -1
            translation_vector[2] *= -1
            camera_positions.append((cam_id, camera_position, rotation_matrix, translation_vector))



            # Scrivi i risultati nel file
            f.write(f"Posizione della camera {cam_id} nel mondo 3D (X, Y, Z in metri):\n")
            f.write(f"{camera_position.flatten()}\n")
            f.write(f"Matrice di rotazione per camera {cam_id}:\n{rotation_matrix}\n")
            f.write(f"Vettore di traslazione per camera {cam_id}:\n{translation_vector.flatten()}\n\n")

    # Visualizzazione 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Disegna il campo con etichette
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
        for col, axis in zip(['r', 'g', 'b'], [R[:, 0], R[:, 1], R[:, 2]]):
            ax.quiver(tvec[0], tvec[1], tvec[2],
                       axis[0], axis[1], axis[2],
                       length=axis_length, color=col, linewidth=2)

    # Disegna tutte le telecamere
    for idx, (cam_id, camera_position, rotation_matrix, translation_vector) in enumerate(camera_positions):
        color = colors[idx % len(colors)]  # Associa un colore unico
        draw_camera(ax, camera_position, rotation_matrix, color, f'Camera {cam_id}')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Ricostruzione 3D del campo di pallavolo\n(Visualizzazione assi telecamere)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Salva l'immagine della visualizzazione 3D
    fig.savefig('camera_positions_3D.png', dpi=300)


if __name__ == "__main__":
    main()
