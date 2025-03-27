import cv2
import json
import os

# Lista delle telecamere da processare
CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
BASE_IMAGE_PATH = 'src-gen/landsmark/out{}-undist.jpg'
BASE_JSON_PATH = 'annotation-undist/out{}-undist.json'
OUTPUT_PATH = 'src-gen/landsmark/out{}-undist-annotated.jpg'

def draw_points_on_image(image_path, json_path, output_path):
    # Carica l'immagine
    image = cv2.imread(image_path)
    if image is None:
        print(f"Errore: immagine non trovata {image_path}")
        return False
    
    # Carica il file JSON
    try:
        with open(json_path) as f:
            points_data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: file JSON non trovato {json_path}")
        return False
    
    # Disegna ogni punto valido
    for point_id, data in points_data.items():
        if data.get("status") == "ok":
            x = int(data["coordinates"]["x"])
            y = int(data["coordinates"]["y"])
            
            # Disegna cerchio e testo
            cv2.circle(image, (x, y), 15, (0, 255, 0), -1)
            text = f"ID:{point_id} ({x},{y})"
            cv2.putText(image, text, (x + 20, y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Salva l'immagine
    cv2.imwrite(output_path, image)
    print(f"Creato: {output_path}")
    return True

# Processa tutte le telecamere
for cam_id in CAMERA_IDS:
    img_path = BASE_IMAGE_PATH.format(cam_id)
    json_path = BASE_JSON_PATH.format(cam_id)
    out_path = OUTPUT_PATH.format(cam_id)
    
    # Crea directory di output se non esiste
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    draw_points_on_image(img_path, json_path, out_path)

print("Elaborazione completata per tutte le telecamere")