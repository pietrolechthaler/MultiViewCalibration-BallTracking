# correct_video.py
import cv2
import numpy as np
import json
import sys

def correct_video(calibration_json, input_video, output_video):
    """
    Correct the distortion of a video using calibration parameters.
    
    :param calibration_json: Path to the calibration JSON file.
    :param input_video: Path to the input video.
    :param output_video: Path to the output video.
    """
    # Load calibration parameters from the JSON file
    with open(calibration_json, 'r') as f:
        calibration_data = json.load(f)

    # Extract the camera matrix (A) and distortion coefficients (D)
    A = np.array(calibration_data['A'])
    D = np.array(calibration_data['D'])

    # Open the input video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video}. Check the file path.")
        return

    # Get video dimensions
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a new camera matrix for distortion correction
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(A, D, (w, h), 1, (w, h))

    # Create a VideoWriter object to save the corrected video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # Read and correct each frame of the video
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if there are no more frames

        # Apply distortion correction
        undistorted_frame = cv2.undistort(frame, A, D, None, new_camera_matrix)

        # Write the corrected frame to the output video
        out.write(undistorted_frame)

        # Print progress
        frame_count += 1
        print(f"Processed frames: {frame_count} of {total_frames}")

    # Release resources
    cap.release()
    out.release()

    print(f"Corrected video saved successfully at: {output_video}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python correct_video.py <calibration_json> <input_video> <output_video>")
        sys.exit(1)

    calibration_json = sys.argv[1]
    input_video = sys.argv[2]
    output_video = sys.argv[3]

    correct_video(calibration_json, input_video, output_video)