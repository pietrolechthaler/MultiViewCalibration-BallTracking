import cv2
import os

video_path = "../../video/prova.mp4"  
output_video_path = "labelImg/data/prova_with_boxes.mp4" 
annotations_folder = "labelImg/data/labels" 

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Errore: Impossibile aprire il video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break 
    frame_name = f"frame_{frame_count:04d}.txt" 
    annotation_path = os.path.join(annotations_folder, frame_name)

    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as file:
            for line in file:
                class_id, x_center, y_center, w, h = map(float, line.split())
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Premi 'q' per uscire
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video con bounding box salvato in: {output_video_path}")