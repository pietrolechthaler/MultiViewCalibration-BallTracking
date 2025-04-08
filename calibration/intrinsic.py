import numpy as np
import cv2
import glob
import os
import pickle
import utils.parameters as parameters
import utils.calibration_utils as utils


def calibrate_camera(folder):
    """
    Calibrate the camera using chessboard images
    Parameters:
        folder (str): Folder containing input images for calibration
    
    Returns:
        ret: The RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """

    print(f"> Processing calibration for images in folder: {folder}")

    # Chessboard configuration
    chessboard_size = utils.read_chessboard_dimensions(folder)
    print(f"> Chessboard size: {chessboard_size}")

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * parameters.SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(os.path.join(folder, '*.jpg'))
    
    if not images:
        print(f"No calibration images found at {folder}")
        return None, None, None, None, None
    
    print(f"> Found {len(images)} calibration images")

    
    images = utils.ranking_images(images, top_n=parameters.TOP_N)
    print(f"> Selected top {len(images)} sharpest images for calibration")
    
    # Create output folder if it doesn't exist
    folder_name = os.path.basename(folder)
    out_name = folder_name + parameters.OUT
    out_path = os.path.join(os.path.dirname(folder), out_name)    

    if not os.path.exists(out_path):
        print(f"> Created output folder: {out_path}")
        os.makedirs(out_path)
        
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            
            # Save image with corners drawn
            output_img_path = os.path.join(out_path, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img)
            
            print(f"- Processed image {idx+1}/{len(images)}: {fname} - Chessboard found")
        else:
            print(f"- Processed image {idx+1}/{len(images)}: {fname} - Chessboard NOT found")
    
    if not objpoints:
        print("- No chessboard patterns were detected in any images.")
        return None, None, None, None, None
    
    print("> Calibrating camera...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret
    }
    
    with open(os.path.join(out_path, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)
    
    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(out_path, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(out_path, 'distortion_coefficients.txt'), dist)
    
    print(f"> Calibration complete! RMS re-projection error: {ret}")
    print(f"> Results saved to {out_path}")
    
    return ret, mtx, dist, rvecs, tvecs


def main():
    """
    Main function to run the camera calibration process.
    """
    print(f"---------------------- Camera Calibration ----------------------")
    
    # Root folder for the generated images
    SRC_GEN = parameters.SRC_GEN

    # Check if the src-gen folder exists
    if not os.path.exists(SRC_GEN):
        print(f"> Error: The folder {SRC_GEN} does not exist.")
        return

    # Process each subfolder in the src-gen folder
    for subfolder_name in os.listdir(SRC_GEN):

        subfolder_path = os.path.join(SRC_GEN, subfolder_name)
        
        if os.path.isdir(subfolder_path):
            # Calibrate the camera
            ret, mtx, dist, rvecs, tvecs = calibrate_camera(subfolder_path)
            
            # Check if calibration was successful
            if mtx is None:
                print("> Calibration failed for folder: {folder_path}")
                continue
            else:
                # Save undistorted images if required
                if(parameters.SAVE_UNDISTORTED):
                    print("> Saving undistorted images...")
                    utils.undistort_images(mtx, dist, subfolder_path)
    
if __name__ == "__main__":
    main()