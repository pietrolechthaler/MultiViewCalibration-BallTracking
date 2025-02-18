"""
Author: Pietro Lechthaler
Description: This script extracts frames from video files for camera calibration purposes.
"""
import cv2
import os
import parameters
import json

def dimensions_to_json(output_path, chessboard_size):
    """
    Write the chessboard dimensions to a JSON file
    @params:
        output_path (str): Path to the output directory for the video.
        chessboard_size (tuple): Dimensions of the chessboard (rows, columns).
    """
    # Create the JSON data
    data = {
        'chessboard_dimensions': {
            'rows': chessboard_size[0], 
            'columns': chessboard_size[1]
        }
    }

    json_path = os.path.join(output_path, parameters.JSON_CHESSBOARD)

    # Write the JSON data to a file    
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

    print(f"- Chessboard dimensions saved to {json_path}")


def find_chessboard(image, grid_size):
    """
    Detect if a chessboard can be found in the image.
    @params:
        image (numpy array): Frame in which to detect the chessboard.
        grid_size (tuple of int): Dimensions of the chessboard to be detected.
    @return:
        bool: True if chessboard is found, False otherwise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCorners(gray, grid_size)[0]

def extract_frames(video_paths, skip_frames, output_folder): 
    """
    Extract frames from each video in the provided list of video paths.
    @params:
        video_paths (list of str): Paths to the video files.
        skip_frames (int): Number of frames to skip between extracts.
        output_folder (str): Directory to save extracted frames.
    """
    # Ensure the base output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each video in the list
    for video, chessboard_size in video_paths:
        
        # Create a unique folder for each video
        video_folder = os.path.join(output_folder, os.path.splitext(os.path.basename(video))[0])
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Open the video
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"> Processing {video}: Total frames in video: {frame_count}")

        frame_idx = 0  # Current frame counter

        while True:
            ret, frame = cap.read()
            # Check if the frame was correctly read
            if not ret:
                break
            
            # Save the frame every 'skip_frames' frames
            if frame_idx % skip_frames == 0 and find_chessboard(frame, chessboard_size):
                frame_file = os.path.join(video_folder, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_file, frame)
                print(f"- Saved {frame_file}")
            
            frame_idx += 1

        # Release the VideoCapture
        cap.release()

        # Create a JSON file with dimensions of the chessboard
        dimensions_to_json(video_folder, chessboard_size)

        print(f"- Finished processing {video} and released video resources.")

"""
Run the script: python3 preprocessing.py
"""
if __name__ == "__main__":
   
    # Parameters
    VIDEO_PATHS = parameters.VIDEO_PATHS
    SKIP_FRAMES = parameters.SKIP_FRAMES
    SRC_GEN = parameters.SRC_GEN

    # Run the function
    extract_frames(VIDEO_PATHS, SKIP_FRAMES, SRC_GEN)