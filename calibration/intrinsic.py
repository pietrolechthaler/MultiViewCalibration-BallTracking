"""
Author: Pietro Lechthaler
Description: This script performs camera calibration using Zhang's method.
"""
import os
import utils.intrinsic_utils as utils
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
def zhang_calibaration(input_folder, checker_size):
    
    name = os.path.basename(input_folder)
    print(f"---------------------- {name} ----------------------")
    print(f"Processing calibration for images in folder: {input_folder}")

    # Chessboard configuration
    chessboard_size = utils.read_chessboard_dimensions(input_folder)
    print(f"> Chessboard size: {chessboard_size}")
    # Create output folder
    out_name = name + parameters.OUT

    # Create output folder
    output_folder = os.path.join(os.path.dirname(input_folder), out_name)    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"> Created output folder: {output_folder}")

    # Load images as a list of Numpy arrays
    images = utils.loadImages(input_folder)
    
    # Detect corners in all images, draw and save them
    imgs_corners = utils.getImagesCorners(images, chessboard_size, output_folder)
    print(f"> Detected corners in {len(imgs_corners)} images")

    # Generate world corners based on the chessboard configuration
    w_corners = utils.getWorldCorners(chessboard_size, checker_size)
    
    print("> Images Corners: ", imgs_corners.shape)
    print("> World Corners: ", w_corners.shape)

    # Calculate the homography matrix for images
    print("> Calculating Homography matrix")
    H_matrices = utils.getAllH(imgs_corners, w_corners)
    
    # Calculate B matrix from the homography matrices
    print("> Calculating B matrix")
    B = utils.getB(H_matrices)
    print("- Estimated B:\n", B)

    # Calculate A (intrinsic parameters) matrix for initialization    
    A_init = utils.getA(B)
    print("> Initialized A (intrinsic parameters):\n", A_init)

    # Calculate rotation and translation matrices for all images
    print("> Calculating all rotation and translation from A")
    RT_all = utils.getRotAndTrans(A_init, H_matrices)
    #K_distortion_init = np.array([0,0])
    K_distortion_init = np.array([0, 0, 0, 0, 0])  # k1, k2, p1, p2, k3
    print("> Initialize the distortion coefficients: ", K_distortion_init)
    
    # Calculate the initial mean error and reprojection errors
    print("> Start optimization")
    print("- Calculating initial mean error and reprojection error")
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
    print("- Optimized A: \n", A_new)
    print("- Radial distortion coefficients (after optimization): ", K_distortion_new)

    # Recalculate errors after optimization
    mean_error_post, reprojected_points = utils.reprojectionRMSError(
        A_new,
        K_distortion_new,
        RT_all,
        imgs_corners,
        w_corners
    )
    mean_error_post = np.round(np.mean(mean_error_post), decimals=6)
    print("- Mean Error (after optimization): ", mean_error_post)

    A_final = x1[0]
    kc_final = x1[1]

    # Extracted from scipy.optimize result
    print('Camera Intrinsic Matrix K:\n', A_new)
    print('\nCamera Distortion Matrix D:\n', K_distortion_new)
    
    # Corrected output to display the camera intrinsic matrix and distortion coefficients
    print('   Focal Length: [ {:.5f}  {:.5f} ]'.format(A_new[0,0], A_new[1,1]))
    print('Principal Point: [ {:.5f}  {:.5f} ]'.format(A_new[0,2], A_new[1,2]))
    print('           Skew: [ {:.7f} ]'.format(A_new[0,1]))
    #print('     Distortion: [ {:.6f}  {:.6f} ]'.format(K_distortion_new[0], K_distortion_new[1]))
    print('     Distortion: [ {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f} ]'.format(K_distortion_new[0], K_distortion_new[1], K_distortion_new[2], K_distortion_new[3], K_distortion_new[4]))

    # Save the calibration results
    utils.saveCalibrationResults(output_folder, name, A_new, K_distortion_new, mean_error_pre, mean_error_post)

"""
Run the script: python3 intrinsic.py
"""
if __name__ == "__main__":

    # Parameters
    SQUARE_SIZE = parameters.SQUARE_SIZE
    SRC_GEN = parameters.SRC_GEN

    # Uncomment to test the calibration for a single folder
    # folder_path = os.path.join(SRC_GEN, "out12F")
    # zhang_calibaration(folder_path, SQUARE_SIZE)
    # sys.exit(0)

    # Check if the src-gen folder exists
    if not os.path.exists(SRC_GEN):
        print(f"Error: The folder {SRC_GEN} does not exist.")
        sys.exit(1) # Exit the script

    # Process each subdirectory in the src-gen folder
    for folder_name in os.listdir(SRC_GEN):
        folder_path = os.path.join(SRC_GEN, folder_name)

        if os.path.isdir(folder_path):
            # Run the calibration for each subdirectory
            zhang_calibaration(folder_path, SQUARE_SIZE)
