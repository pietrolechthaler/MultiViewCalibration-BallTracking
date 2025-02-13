"""
Author: Pietro Lechthaler
Description: This script extracts frames from video files for camera calibration purposes.
"""
import cv2
import os
import parameters
"""
Extract frames from each video in the provided list of video paths.
@params:
    video_paths (list of str): Paths to the video files.
    skip_frames (int): Number of frames to skip between extracts.
    output_folder (str): Directory to save extracted frames.
"""
def extract_frames(video_paths, skip_frames, output_folder): 
    # Ensure the base output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each video in the list
    for video in video_paths:
        # Create a unique folder for each video
        video_folder = os.path.join(output_folder, os.path.splitext(os.path.basename(video))[0])
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Open the video
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {video}: Total frames in video: {frame_count}")

        frame_idx = 0  # Current frame counter

        while True:
            ret, frame = cap.read()
            # Check if the frame was correctly read
            if not ret:
                break
            
            # Save the frame every 'skip_frames' frames
            if frame_idx % skip_frames == 0:
                frame_file = os.path.join(video_folder, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_file, frame)
                print(f"Saved {frame_file}")
            
            frame_idx += 1

        # Release the VideoCapture
        cap.release()
        print(f"Finished processing {video} and released video resources.")

"""
Run the script: python3 extract_frames.py
"""
if __name__ == "__main__":
   
    # Parameters
    VIDEO_PATHS = parameters.VIDEO_PATHS
    SKIP_FRAMES = parameters.SKIP_FRAMES
    SRC_GEN = parameters.SRC_GEN

    # Run the function
    extract_frames(VIDEO_PATHS, SKIP_FRAMES, SRC_GEN)