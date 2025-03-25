"""
Author: Pietro Lechthaler
Description: 
"""
import json
import os
import utils.parameters as parameters
import glob
import cv2


def read_chessboard_dimensions(folder):
    """
    Read the chessboard dimensions from a JSON file.
    @params:
        json_path (str): Path to the folder containing the json with chessboard dimensions.
    @return:
        tuple: Tuple containing the dimensions of the chessboard (rows,, columns)
    """
    # Load the JSON file
    filename = parameters.JSON_CHESSBOARD
    json_path = os.path.join(folder, filename)

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        
    rows = data['chessboard_dimensions']['rows']
    columns = data['chessboard_dimensions']['columns']

    return (rows, columns)

def undistort_images(mtx, dist, folder):
    """
    Undistort all calibration images using the calibration results.
    
    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
        folder: Directory containing calibration images
    """
    
    images = glob.glob(os.path.join(folder, '*.jpg'))
    
    if not images:
        print(f"No images found at {folder}")
        return
    
    # Create output folder if it doesn't exist
    name = os.path.basename(folder)
    out_name = name + parameters.DISTORTION
    undistorted_dir = os.path.join(os.path.dirname(folder), out_name)    

    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)
    
    print(f"> Undistorting {len(images)} images...")
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]
        
        # Refine camera matrix based on free scaling parameter
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # Undistort image
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        
        # Crop the image (optional)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        
        # Save undistorted image
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)
        
        print(f"- Undistorted image {idx+1}/{len(images)}: {fname}")
    
    print(f"- Undistorted images saved to {undistorted_dir}")