import cv2

video_path = "../../video/out1.mp4"
output_folder = "frames2"
cap = cv2.VideoCapture(video_path)
frame_count = 480

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.png", frame)
    print(f"Frame {frame_count} salvato")
    frame_count += 1

cap.release()