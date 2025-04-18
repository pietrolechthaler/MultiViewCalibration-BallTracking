from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
import sys

# Constants
TRAIN_PATH = 'tracking/runs/detect/train4/'
VIDEO_SUBDIR = 'video/'
COORDS_DIR = 'tracking/coordinates/'
START_SEC = 155
END_SEC = 166 

def setup_directories(base_path):
    """Create necessary directories if they don't exist"""
    video_dir = Path(base_path) / VIDEO_SUBDIR
    coords_dir = COORDS_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, coords_dir

def process_video(input_path, model, video_dir, coords_dir, camera_id, start_sec, end_sec):
    """Process a single video file within a specified time range (seconds)"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = video_dir / f"out{camera_id}_detected.mp4"
    output_csv_path = coords_dir / f"out{camera_id}_coordinates.csv"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    with open(output_csv_path, 'w') as csv_file:
        csv_file.write("timestamp_sec,x_center,y_center,confidence\n")

        while True:
            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            current_sec = current_msec / 1000.0
            if current_sec > end_sec:
                break

            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.30)
            result = results[0]

            if len(result.boxes) > 0:
                best_idx = result.boxes.conf.cpu().numpy().argmax()
                box = result.boxes.xyxy.cpu().numpy()[best_idx]
                confidence = result.boxes.conf.cpu().numpy()[best_idx]

                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                csv_file.write(f"{current_sec:.2f},{x_center:.2f},{y_center:.2f},{confidence:.4f}\n")

                x1, y1, x2, y2 = map(int, box)
                label = f"Volleyball {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {output_video_path}")
    print(f"Coordinates saved to: {output_csv_path}")

def main(camera_id, start_sec, end_sec):
    try:
        video_dir, coords_dir = setup_directories(TRAIN_PATH)

        model_path = Path(TRAIN_PATH) / 'weights/best_v11_800.pt'
        model = YOLO(str(model_path))

        input_path = f"video/match/out{camera_id}.mp4"

        process_video(input_path, model, video_dir, coords_dir, camera_id, start_sec, end_sec)

    except Exception as e:
        print(f"Error processing camera {camera_id}: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_id', type=int, help='Camera ID to process')
    args = parser.parse_args()

    main(args.camera_id, START_SEC, END_SEC)
