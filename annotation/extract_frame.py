#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np

# Configuration
INPUT_DIR = "../video/match"
OUTPUT_DIR = "../src-gen/landsmark"
CALIB_ROOT = "../src-gen"

CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

def load_calibration_files(calib_folder):
    """
    Load calibration data from folder containing calibration files
    Returns: mtx (camera matrix), dist (distortion coefficients)
    """
    try:
        mtx = np.loadtxt(os.path.join(calib_folder, 'camera_matrix.txt'))
        dist = np.loadtxt(os.path.join(calib_folder, 'distortion_coefficients.txt'))
        return mtx, dist
    except Exception as e:
        print(f"Error loading calibration: {str(e)}")
        return None, None

def undistort_single_image(mtx, dist, img_path, output_path):
    """
    Apply undistortion to single image and save the result.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        h, w = img.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, new_mtx)
        
        # Crop to ROI
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        cv2.imwrite(output_path, undistorted)
        print(f"Undistorted image saved to {output_path}")
        return True
    except Exception as e:
        print(f"Undistortion failed: {str(e)}")
        return False

def extract_frame(video_path, time_sec, output_path):
    """Extract a frame from video at specified time and save it"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_pos = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame at {time_sec}s from {video_path}")
        return False
    
    cv2.imwrite(output_path, frame)
    return True

def process_single_video(i, time_sec):
    """Process a single video with given index and time point"""
    # Prepare paths
    calib_folder = os.path.join(CALIB_ROOT, f"out{i}F-gen")
    video_path = os.path.join(INPUT_DIR, f"out{i}.mp4")  # Assuming video has no extension
    input_img = os.path.join(OUTPUT_DIR, f"out{i}.jpg")
    output_img = os.path.join(OUTPUT_DIR, f"out{i}-undist.jpg")
    
    # Check if calibration folder exists
    if not os.path.isdir(calib_folder):
        print(f"Skipping out{i} - missing calibration folder {calib_folder}")
        return False
    
    # Extract frame from video if it doesn't exist
    if not os.path.exists(input_img):
        if not extract_frame(video_path, time_sec, input_img):
            return False
    
    # Check if input image exists
    if not os.path.isfile(input_img):
        print(f"Skipping out{i} - missing input image {input_img}")
        return False
    
    print(f"Processing out{i} with calibration from {calib_folder}")
    
    # Load calibration data
    mtx, dist = load_calibration_files(calib_folder)
    if mtx is None or dist is None:
        return False
    
    # Apply undistortion
    return undistort_single_image(mtx, dist, input_img, output_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('i', type=int)
    parser.add_argument('time', type=float)
    
    args = parser.parse_args()
    
    if not args.i in CAMERA_IDS:
        print("Error: camera ID must be one of the following: ", CAMERA_IDS)
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process the video
    success = process_single_video(args.i, args.time)
    
    if success:
        print("Undistortion process completed successfully")
        sys.exit(0)
    else:
        print("Undistortion process failed")
        sys.exit(1)

if __name__ == "__main__":
    main()