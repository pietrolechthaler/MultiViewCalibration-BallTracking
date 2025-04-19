from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
import sys

from utils.parameters import TRAIN_PATH, VIDEO_SUBDIR, COORDS_DIR, START_SEC, END_SEC, WEIGHTS

def setup_directories(base_path):
    video_dir = Path(base_path) / VIDEO_SUBDIR
    coords_dir = COORDS_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, coords_dir


def select_roi(frame, max_size=1000):
    h, w = frame.shape[:2]
    scale = 1.0

    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        resized_frame = frame

    print("Select ROI and press ENTER or SPACE when done, or press 'c' to cancel.")
    roi = cv2.selectROI("Select ROI", resized_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    x, y, w_roi, h_roi = roi
    x = int(x / scale)
    y = int(y / scale)
    w_roi = int(w_roi / scale)
    h_roi = int(h_roi / scale)

    return (x, y, w_roi, h_roi)


def sliding_window_detection(frame, model, roi_coords, window_size=(640, 640), overlap=0.1):
    x_roi, y_roi, w_roi, h_roi = roi_coords
    best_confidence = -1
    best_box = None

    step_x = int(window_size[0] * (1 - overlap))
    step_y = int(window_size[1] * (1 - overlap))

    for y in range(y_roi, y_roi + h_roi, step_y):
        for x in range(x_roi, x_roi + w_roi, step_x):
            x_end = min(x + window_size[0], x_roi + w_roi)
            y_end = min(y + window_size[1], y_roi + h_roi)

            window = frame[y:y_end, x:x_end]

            if window.size == 0:
                continue

            results = model.predict(source=window, conf=0.30)
            result = results[0]

            if len(result.boxes) > 0:
                idx = result.boxes.conf.cpu().numpy().argmax()
                box = result.boxes.xyxy.cpu().numpy()[idx]
                confidence = result.boxes.conf.cpu().numpy()[idx]

                x1, y1, x2, y2 = box
                x1 += x
                x2 += x
                y1 += y
                y2 += y

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = (x1, y1, x2, y2, best_confidence)

    return best_box


def process_video(input_path, model, video_dir, coords_dir, camera_id, start_sec, end_sec):
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

    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Cannot read first frame for ROI selection.")

    roi = select_roi(first_frame)
    x_roi, y_roi, w_roi, h_roi = roi

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

            if camera_id in [2, 12, 13]:
                best_detection = sliding_window_detection(frame, model, (x_roi, y_roi, w_roi, h_roi))
                if best_detection:
                    x1, y1, x2, y2, confidence = best_detection

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    csv_file.write(f"{current_sec:.2f},{x_center:.2f},{y_center:.2f},{confidence:.4f}\n")

                    label = f"Volleyball {confidence:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            else:
                roi_frame = frame[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
                results = model.predict(source=roi_frame, conf=0.30)
                result = results[0]

                if len(result.boxes) > 0:
                    best_idx = result.boxes.conf.cpu().numpy().argmax()
                    box = result.boxes.xyxy.cpu().numpy()[best_idx]
                    confidence = result.boxes.conf.cpu().numpy()[best_idx]

                    x1, y1, x2, y2 = box
                    x1 += x_roi
                    x2 += x_roi
                    y1 += y_roi
                    y2 += y_roi

                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    csv_file.write(f"{current_sec:.2f},{x_center:.2f},{y_center:.2f},{confidence:.4f}\n")

                    label = f"Volleyball {confidence:.2f}"
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {output_video_path}")
    print(f"Coordinates saved to: {output_csv_path}")


def main(camera_id, start_sec, end_sec):
    try:
        video_dir, coords_dir = setup_directories(TRAIN_PATH)

        model_path = Path(TRAIN_PATH) / WEIGHTS
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