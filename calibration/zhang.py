"""
Author: Pietro Lechthaler
Description: This script performs camera calibration using Zhang's method.
"""
import os
import utils.zhang_utils as utils
from tqdm import tqdm
import utils.parameters as parameters
import sys
import numpy as np
import scipy

"""
Perform camera calibration using Zhang's method
@params:
    input_folder (str): Directory containing input images for calibration
    checker_size (int): Size of the chessboard in millimiters
    chessBoardSize (tuple): Number of inner corners in the chessboard
"""
def zhang_calibaration(input_folder, checker_size=28, chessboard_size=(7,5)):
   
    name = os.path.basename(input_folder)
    print(f"---------------------- {name} ----------------------")
    print(f"Processing calibration for images in folder: {input_folder}")

    # Create output folder
    out_name = name + parameters.OUT

    # Create output folder
    output_folder = os.path.join(os.path.dirname(input_folder), out_name)    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"> Created output folder: {output_folder}")

    # Load images as a list of Numpy arrays
    images = utils.loadImages(input_folder)
    
    # Detect corners in all images
    imgs_corners = utils.getImagesCorners(images, chessboard_size, output_folder)

    # Generate world corners based on the chessboard configuration
    w_corners = utils.getWorldCorners(chessboard_size, checker_size)
    
    print("\n> Images Corners: ", imgs_corners.shape)
    print("> World Corners: ", w_corners.shape)

    # Calculate the homography matrix for images
    print("> Calculating Homography matrix")
    H_matrices = utils.getAllH(imgs_corners, w_corners)
    
    # Calculate B matrix from the homography matrices
    print("> Calculating B matrix")
    B = utils.getB(H_matrices)
    print("- Estimated B: ", B)

    # Calculate A (intrinsic parameters) matrix for initialization    
    A_init = utils.getA(B)
    print("\n> Initialized A (intrinsic parameters): ", A_init)

    # Calculate rotation and translation matrices for all images
    print("\n> Calculating all images and translation matrices (extrinsics parameters)")
    RT_all = utils.getRotAndTrans(A_init, H_matrices)
    K_distortion_init = np.array([0,0])
    print("> Initialize the radial distortion parameters: ", K_distortion_init)
    
    # Calculate the initial mean error and reprojection errors
    print("> Calculating initial mean error and reprojection error")
    mean_error_pre, reprojected_points = utils.reprojectionRMSError(
        A_init,
        K_distortion_init,
        RT_all,
        imgs_corners,
        w_corners
    )

    mean_error_pre = np.round(np.mean(mean_error_pre), decimals=6)
    print("- Mean Error (before optimization): ", mean_error_pre)

    # Perform the optimization
    print("> Running Least Square Optimization")
    x0 = utils.extractParamFromA(A=A_init, K_distortion_init=K_distortion_init) # initial parameters

    res = scipy.optimize.least_squares(
        fun= utils.loss_func,
        x0=x0,
        method="lm",
        args=[RT_all, imgs_corners, w_corners],
    )

    # Extract new camera parameters after optimization
    x1 = res.x
    A_new, K_distortion_new = utils.retrieveA(x1)
    print("- Optimized A: ", A_new)
    print("- Radial distortion parameters (after optimization): ", K_distortion_new)

    # Recalculate errors after optimization
    print("\n> Calculating Initial mean error and reprojection error")
    mean_error_post, reprojected_points = utils.reprojectionRMSError(
        A_new,
        K_distortion_new,
        RT_all,
        imgs_corners,
        w_corners
    )
    mean_error_post = np.round(np.mean(mean_error_post), decimals=6)
    print("- Mean Error (after optimization): ", mean_error_post)

    # Save the calibration results
    print("\n> Saving calibration results")
    utils.saveCalibrationResults(output_folder, name, A_new, RT_all, K_distortion_new, mean_error_pre, mean_error_post)

"""
Run the script: python3 zhang.py <video_paths> <skip_frames> <output_folder>
"""
if __name__ == "__main__":

    # Parameters
    CHESS_BOARD_SIZE = parameters.CHESS_BOARD_SIZE
    CHECKER_SIZE = parameters.CHECKER_SIZE
    SRC_GEN = parameters.SRC_GEN
    
    # Check if the src-gen folder exists
    if not os.path.exists(SRC_GEN):
        print(f"Error: The folder {SRC_GEN} does not exist.")
        sys.exit(1) # Exit the script

    # Process each subdirectory in the src-gen folder
    for folder_name in os.listdir(SRC_GEN):
        folder_path = os.path.join(SRC_GEN, folder_name)

        if os.path.isdir(folder_path):
            # Run the calibration for each subdirectory
            zhang_calibaration(folder_path, CHECKER_SIZE, CHESS_BOARD_SIZE)
