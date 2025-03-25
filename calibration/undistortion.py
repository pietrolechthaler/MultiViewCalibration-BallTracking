import cv2
import numpy as np
import os
import sys
import pickle

def load_calibration_files(calib_folder):
    """
    Load calibration data from folder containing:
    - calibration_data.pkl
    - camera_matrix.txt
    - distortion_coefficients.txt
    
    Returns:
        mtx: Camera matrix
        dist: Distortion coefficients
    """
    try:
        # Load from text files (primary source)
        mtx = np.loadtxt(os.path.join(calib_folder, 'camera_matrix.txt'))
        dist = np.loadtxt(os.path.join(calib_folder, 'distortion_coefficients.txt'))
        
        # Verify pkl exists for consistency check
        pkl_path = os.path.join(calib_folder, 'calibration_data.pkl')
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, using text files only")
            
        return mtx, dist
        
    except Exception as e:
        print(f"Error loading calibration: {str(e)}")
        return None, None

def undistort_single_image(mtx, dist, img_path, output_path):
    """
    Apply undistortion to single image using OpenCV's optimal parameters
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        h, w = img.shape[:2]
        
        # Get optimal new camera matrix (matches calibration approach)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h)
        )
        
        # Apply undistortion (same method as in reference)
        undistorted = cv2.undistort(
            img, mtx, dist, None, new_mtx
        )
        
        # Crop to ROI (matches reference implementation)
        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]
        
        cv2.imwrite(output_path, undistorted)
        print(f"Undistorted image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Undistortion failed: {str(e)}")
        return False

if __name__ == "__main__":

    
    calib_folder = sys.argv[1]
    input_img = sys.argv[2]
    output_img = sys.argv[3]
    
    # Load calibration
    mtx, dist = load_calibration_files(calib_folder)
    if mtx is None or dist is None:
        sys.exit(1)
    
    # Process image
    success = undistort_single_image(mtx, dist, input_img, output_img)
    sys.exit(0 if success else 1)
