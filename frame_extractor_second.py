import cv2

def get_frame_from_video(video_path, time_in_seconds):
    # Apri il video
    cap = cv2.VideoCapture(video_path)

    # Controlla se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'aprire il video.")
        return None, None

    # Ottieni FPS e calcola il frame corrispondente
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int(fps * time_in_seconds)

    # Imposta il video al frame desiderato
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Leggi il frame
    success, frame = cap.read()
    cap.release()

    if success:
        return frame_number, frame
    else:
        print("Errore nel recupero del frame.")
        return None, None

# Esempio di utilizzo
video_path = './video/match/out1.mp4'  # Sostituisci con il tuo path video
time_in_seconds = 155.92  # Inserisci i secondi come float

frame_number, frame = get_frame_from_video(video_path, time_in_seconds)

if frame is not None:
    print(f"Frame numero: {frame_number}")
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Salva il frame come immagine
    cv2.imwrite(f'frame_{frame_number}-1.jpg', frame)
else:
    print("Nessun frame trovato.")
