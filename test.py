import numpy as np
import cv2
import pickle

def calculate_homography(K_i, R_i, t_i, K_j, R_j, t_j):
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i
    H_ij = K_j @ (R_rel - t_rel @ np.linalg.inv(K_i))
    return H_ij

# Funzione per caricare i dati di calibrazione
def load_calibration_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

try:
    # Carica i dati di calibrazione delle camere
    data1 = load_calibration_data('./src-gen/out1F-gen/calibration_data.pkl')
    data11 = load_calibration_data('./src-gen/out1F-gen/calibration_extrinsic.pkl')
    data2 = load_calibration_data('./src-gen/out3F-gen/calibration_data.pkl')
    data22 = load_calibration_data('./src-gen/out3F-gen/calibration_extrinsic.pkl')

    # Estrai le matrici di calibrazione
    K_i = data1['camera_matrix']
    R_i = data11['rotation_matrix']
    t_i = data11['translation_vector']

    K_j = data2['camera_matrix']
    R_j = data22['rotation_matrix']
    t_j = data22['translation_vector']

    # Calcola l'omografia
    H_ij = calculate_homography(K_i, R_i, t_i, K_j, R_j, t_j)
    print(f"Homografia H_ij:\n{H_ij}")

    # Definisci le dimensioni del campo stilizzato
    campo_larghezza_m = 9.0  # larghezza del campo in metri
    campo_lunghezza_m = 18.0  # lunghezza del campo in metri

    # Definisci la dimensione dell'immagine stilizzata
    campo_larghezza_px = 800  # larghezza in pixel
    campo_lunghezza_px = 1600  # lunghezza in pixel

    # Calcola la scala
    scala = campo_larghezza_px / campo_larghezza_m  # pixel per metro

    # Punto nel campo stilizzato (esempio)
    P_campo = np.array([300, 200])  # Coordinate del punto nel campo in pixel

    # Converti le coordinate del campo in metri
    P_campo_m = np.array([P_campo[0] / scala, P_campo[1] / scala])  # Converti in metri

    # Converti in coordinate omogenee
    P_campo_homogeneo = np.array([P_campo_m[0], P_campo_m[1], 1])

    # Proietta il punto nell'immagine della camera j
    P_j = H_ij @ P_campo_homogeneo
    P_j_norm = P_j[:2] / P_j[2]  # Normalizzazione

    # Carica l'immagine della camera j
    image_j = cv2.imread('./annotation/annotation-images/out3.jpg')
    if image_j is None:
        raise FileNotFoundError("L'immagine non è stata trovata nel percorso specificato.")

    # Disegna il punto proiettato sull'immagine
    cv2.circle(image_j, (int(P_j_norm[0]), int(P_j_norm[1])), radius=5, color=(0, 255, 0), thickness=-1)

    # Mostra l'immagine
    cv2.imshow('Proiezione Punto', image_j)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"Errore: {e}")
