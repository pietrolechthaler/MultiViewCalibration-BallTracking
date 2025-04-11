from ultralytics import YOLO
import cv2
import json
from pathlib import Path
import argparse
import sys

def detection(i):
    # Carica il modello YOLO
    model = YOLO('runs/detect/train4/weights/best.pt')

    # Percorso del video di input
    input_video_path = f"../video/match/out{i}.mp4"
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Errore: Impossibile aprire il video {input_video_path}")
        sys.exit(1)

    # Ottieni le proprietà del video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Percorso del video di output
    output_video_path = f"detection/out{i}_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Percorso del file JSON di output
    output_json_path = f"detection/out{i}_coordinates.json"
    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)

    # Lista per salvare i dati JSON
    json_data = []

    results = model.predict(source=input_video_path, stream=True)

    for frame_idx, result in enumerate(results):
        frame = result.orig_img
        boxes = result.boxes.xyxyn.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for ball_idx, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            
            json_data.append({
                "frame_id": frame_idx,
                "ball_id": ball_idx,
                "x_center": float(x_center),
                "y_center": float(y_center),
                "confidence": float(confidence)
            })
            
            x1, y1, x2, y2 = map(int, box * [frame_width, frame_height, frame_width, frame_height])
            label = f"Volleyball {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        out.write(frame)

    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video salvato in: {output_video_path}")
    print(f"Coordinate salvate in: {output_json_path}")

if __name__ == "__main__":

    try:
        i = int(sys.argv[1])
    except ValueError:
        print("Errore: id camera non valido")
        sys.exit(1)
    
    detection(i)