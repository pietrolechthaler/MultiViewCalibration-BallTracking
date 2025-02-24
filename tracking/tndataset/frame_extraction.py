import cv2

video_path = "../../video/prova.mp4"
output_folder = "labelImg/data/frames"
cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.png", frame)
    frame_count += 1

cap.release()