from ultralytics import YOLO
import cv2
import numpy as np


model = YOLO('runs/detect/train/weights/best.pt')

# zona di esclusione (x1, y1, x2, y2)
exclude_zone = (790, 180, 820, 200)  


def is_in_exclude_zone(box, exclude_zone):
    x1, y1, x2, y2 = box
    ex1, ey1, ex2, ey2 = exclude_zone
    return not (x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2)


input_video_path = '../video/out1.mp4'
cap = cv2.VideoCapture(input_video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_video_path = 'out1_filtered.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per MP4
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


results = model.predict(source=input_video_path, stream=True)

for frame_idx, result in enumerate(results):
    frame = result.orig_img

    boxes = result.boxes.xyxy.cpu().numpy()  # Coordinate delle bounding box
    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

    filtered_boxes = []
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        if class_id == 0: #and confidence > 0.6:  
            if not is_in_exclude_zone(box, exclude_zone):
                filtered_boxes.append(box)


    for box in filtered_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Colore verde, spessore 2

    out.write(frame)


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video salvato in: {output_video_path}")