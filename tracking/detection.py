from ultralytics import YOLO
import cv2
from pathlib import Path
import argparse
import sys

# Constants
TRAIN_PATH = 'tracking/runs/detect/train4/'
VIDEO_SUBDIR = 'video/'
COORDS_SUBDIR = 'coordinates/'

def setup_directories(base_path):
    """Create necessary directories if they don't exist"""
    video_dir = Path(base_path) / VIDEO_SUBDIR
    coords_dir = Path(base_path) / COORDS_SUBDIR
    video_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, coords_dir

def process_video(input_path, model, video_dir, coords_dir, camera_id):
    """Process a single video file"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Prepare output paths
    output_video_path = video_dir / f"out{camera_id}_detected.mp4"
    output_csv_path = coords_dir / f"out{camera_id}_coordinates.csv"

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))

    # Process video frames
    results = model.predict(source=input_path, stream=True, conf=0.4)

    with open(output_csv_path, 'w') as csv_file:
        csv_file.write("frame_idx,ball_idx,x_center,y_center,confidence\n")
        
        for frame_idx, result in enumerate(results):
            frame = result.orig_img
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for ball_idx, (box, confidence) in enumerate(zip(boxes, confidences)):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                
                # Write to CSV
                csv_file.write(f"{frame_idx},{ball_idx},{x_center},{y_center},{confidence}\n")
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box * [frame_width, frame_height, frame_width, frame_height])
                label = f"Volleyball {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

            out.write(frame)

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to: {output_video_path}")
    print(f"Coordinates saved to: {output_csv_path}")

def main(camera_id):
    try:
        video_dir, coords_dir = setup_directories(TRAIN_PATH)
        
        # Load YOLO model
        model_path = Path(TRAIN_PATH) / 'weights/best.pt'
        model = YOLO(str(model_path))
        
        # Process video
        input_path = f"video/match/out{camera_id}.mp4"
        process_video(input_path, model, video_dir, coords_dir, camera_id)
        
    except Exception as e:
        print(f"Error processing camera {camera_id}: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('camera_id', type=int, help='Camera ID to process')
    args = parser.parse_args()
    
    main(args.camera_id)