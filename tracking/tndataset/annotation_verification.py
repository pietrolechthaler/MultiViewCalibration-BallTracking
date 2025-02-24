import cv2
import os

# Percorsi dei file
video_path = "../../video/prova.mp4"  
output_video_path = "labelImg/data/prova_with_boxes.mp4"  # Percorso del video di output
annotations_folder = "labelImg/data/labels"  # Cartella contenente i file .txt delle annotazioni

# Apri il video originale
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Errore: Impossibile aprire il video.")
    exit()

# Ottieni le informazioni del video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crea un oggetto VideoWriter per salvare il video con le bounding box
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il formato MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    # Leggi il frame successivo
    ret, frame = cap.read()
    if not ret:
        break  # Fine del video

    # Costruisci il percorso del file di annotazione per questo frame
    frame_name = f"frame_{frame_count:04d}.txt"  # Formato: frame_0001.txt, frame_0002.txt, ecc.
    annotation_path = os.path.join(annotations_folder, frame_name)

    # Se esiste un file di annotazione per questo frame, disegna le bounding box
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as file:
            for line in file:
                class_id, x_center, y_center, w, h = map(float, line.split())
                # Converti le coordinate normalizzate in pixel
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                # Calcola i vertici del rettangolo
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                # Disegna il rettangolo sull'immagine
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Scrivi il frame nel video di output
    out.write(frame)

    # Mostra il frame (opzionale)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Premi 'q' per uscire
        break

    frame_count += 1

# Rilascia le risorse
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video con bounding box salvato in: {output_video_path}")