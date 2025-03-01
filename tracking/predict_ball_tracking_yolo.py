from ultralytics import YOLO
import cv2

# Carica il modello YOLO
model = YOLO('runs/detect/train3/weights/best.pt')

# Percorso del video di input
input_video_path = '../video/out1.mp4'
cap = cv2.VideoCapture(input_video_path)

# Ottieni le proprietà del video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Percorso del video di output
output_video_path = 'detection/out1_detected.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

results = model.predict(source=input_video_path, stream=True)

for result in results:
    frame = result.orig_img  # Frame originale
    boxes = result.boxes.xyxy.cpu().numpy()  # Coordinate delle bounding box
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  # Converti le coordinate in interi
        label = f"Volleyball {confidence:.2f}" 

        # Disegna il rettangolo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video salvato in: {output_video_path}")