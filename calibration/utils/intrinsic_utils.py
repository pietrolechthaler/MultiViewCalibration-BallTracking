"""
Author: Pietro Lechthaler
Description: 
"""
import datetime
import json
import cv2
import numpy as np
import os
import copy
import utils.parameters as parameters
import sys
def loadImages(folder):
    """
    Load images from a given folder path into a list of numpy arrays
    @params: 
        folder (str): Folder path containing the images
    @return:
        images (List[np.ndarray]): List of images represented as numpy arrays
    """

    # List all files in the folder
    files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    images = []
    for filename in files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Failed to load image: {filename}")

    return images

def getImagesCorners(images, chessBoardSize, output_folder):
    """
    Extracts the corners of a chessboard pattern from a list of images.
    
    This function iterates through each image, attempting to find the corners of a chessboard pattern
    defined by `chessBoardSize_xy`. If corners are found, they are reshaped and added to a list.
    Each set of corners is also drawn on the image for visualization.
    
    Args:
        images (List[np.ndarray]): List containing the images in which to find chessboard corners
        chessBoardSize (tuple): Dimensions (columns, rows) of the chessboard pattern (excluding the edges)
        output_folder (str): Folder path to save the images with detected corners
    
    Returns:
        output_corners (np.ndarray): Array containing the detected corners
    """

    # List to store the corners of each image
    output_corners = []
    counter = 1
    
    # Iterate through each image
    for img in images:
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(image=img, patternSize=chessBoardSize)

        # Corners detected
        if ret == True:
            # Reshape the corners into 2D and add them to the list
            img_corners = corners.reshape(-1, 2)            
            output_corners.append(img_corners)

            # Draw the corners on the image for visualization
            drawCorners(img, img_corners, counter, output_folder)
            counter += 1
        
    return np.array(output_corners)


def drawCorners(img, corners, counter, output_folder):
    """
    Draws circles on the corners detected in an image and saves the annotated image to a file
    @params:
        img (np.array): Input image on which to draw corners
        corners (List): List of detected corners in the image, each corner is represented as (x, y)
        counter (int): Counter to keep track of the image number
        output_folder (str): Folder path to save the images with detected corners
    """

    # Create a copy of the image to draw on
    out_img = copy.deepcopy(img)

    # Draw circles on corners
    for i in range(len(corners)):
        cv2.circle(img=out_img, 
                   center=(int(corners[i][0]), int(corners[i][1])), 
                   radius=7, 
                   color=(0,0,255), 
                   thickness=-1)

    # Output file path    
    out_path = os.path.join(output_folder, "calib_" + str(counter) + ".jpg")
    
    # Save the image with detected corners
    cv2.imwrite(out_path, out_img)
    print(f"- Saved image with detected corners to: {out_path}")


def getWorldCorners(chessboard_size, checker_size):
    """
    Generates the 2D world coordinates of corners present in the printed Chessboard pattern
    It assumes that the chessboard is oriented such that one corner is at the origin (0,0) 
    and extends in the positive x and y directions.
    
    @params:
        chessboard_size (tuple): Dimensions of the chessboard as (columns, rows)
        checker_size (float): The length of one side of a square (checker) on the chessboard

    @return:
        world_corners (np.ndarray): Array containing the 2D world coordinates of the corners
    """

    world_corners = []
    
    rows =  chessboard_size[1] + 1 # Number of rows
    columns = chessboard_size[0] + 1 # Number of columns
    
    # Loop over each row (y-direction)
    for i in range(1, rows):
        
        # Loop over each column (x-direction)
        for j in range(1, columns):
            # Calculate the coordinates for each corner:
            # multiply by checker_size to scale the coordinates by the size of the squares
            world_corners.append((i*checker_size, j*checker_size))

    return np.array(world_corners)


def getH(img_corners, world_corners):
    """
    Computes the homography matrix from corresponding points in image and world coordinates

    This function uses DLT method. It constructs a matrix from the coordinates, 
    applies SVD to solve for the homography matrix that maps the world coordinates 
    (chessboard pattern) to the image coordinates.

    @params:
        img_corners (np.ndarray): Corners detected in the image
        world_corners(np.ndarray): Predefined corners in the world plane
    
    @returns:
        np.ndarray: 3x3 homography matrix that transforms world coordinates to image coordinates
    """
    # Initialize a list to store the rows of the matrix used in the DLT algorithm
    h = []

    # Check if the number of points in image corners matches the number of points in world corners
    if np.shape(img_corners) == np.shape(world_corners):

        # Construct the matrix 'A' from the corresponding points
        for i in range(len(img_corners)):
            x_i, y_i = img_corners[i] # image coords
            X_i, Y_i = world_corners[i] # world coords

            # Each point pair contributes two rows to matrix 'A'
            row1 = np.array([-X_i, -Y_i, -1, 0, 0, 0, x_i*X_i, x_i*Y_i, x_i])
            row2 = np.array([0, 0, 0, -X_i, -Y_i, -1, y_i*X_i, y_i*Y_i, y_i])
            h.extend([row1, row2])

    # apply SVD to get the homography matrix
    h = np.array(h)
    U, S, V_T = np.linalg.svd(h)
    H = V_T[-1].reshape(3, 3)  # Use the last row of V_T and reshape to 3x3
    H /= H[-1, -1]  # Normalize so that the bottom-right value is 1

    return H


def getAllH(images_corners, world_corners):
    """
    Calculates the homography matrices for all images based on detected corners in the images and the predefined world corners
    
    @params:
        img_corners (np.ndarray):   Array of corners detected in each image, 
                                    structured as [num_images, num_corners, 2]
        world_corners(np.ndarray):  Array of predefined corners on the world plane (chessboard),
                                    structured as [num_corners, 2]
    @returns:
        np.ndarray: An array of homography matrices, one for each image, structured as [img_count, 3, 3]
    """
    # List to store all homography matrices
    H_matrices = []

    # Loop through each set of image corners and calculate the homography matrix
    for img_corners in images_corners:

        # Compute the homography matrix for current set of image corners and the world corners
        H = getH(img_corners, world_corners)
        H_matrices.append(H)
    
    # Convert the list of homography matrices to a numpy array
    return np.array(H_matrices)


def getVij(hi: np.ndarray, hj: np.ndarray)-> np.ndarray:
    """
    hi : ith column of matrix H
    hj: jth column of matri H
    """
    Vij = np.array(
        [
            hi[0]* hj[0],
            hi[0]* hj[1] + hi[1]*hj[0],
            hi[1]*hj[1],
            hi[2]*hj[0] + hi[0]*hj[2],
            hi[2]*hj[1] + hi[1]*hj[2],
            hi[2]*hj[2]
        ]
    )

    return Vij.T

def getV(H_all: np.ndarray) -> np.ndarray:
    v = []

    for H in H_all:
        h1 = H[:, 0] # first column of H matrix
        h2 = H[:, 1] # second column of H matrix

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11-v22).T)
    
    return np.array(v) # shape (2*images, 6)


def arrangeB(b):
    B = np.zeros((3,3))
    B[0, 0] = b[0]
    B[0, 1] = b[1]
    B[0, 2] = b[3]
    B[1, 0] = b[1]
    B[1, 1] = b[2]
    B[1, 2] = b[4]
    B[2, 0] = b[3]
    B[2, 1] = b[4]
    B[2, 2] = b[5]

    return B

def getB(H_all):
    """
    Computes the matrix B from a collection of homography matrices.

    @params:
        H_all (np.ndarray): An array containing multiple homography matrices, each of shape (3, 3)

    @returns:
        np.ndarray: Matrix B derived from the homography matrices
    """
    # Compute the matrix v from the homography matrices
    v = getV(H_all)
    
    # Apply Singular Value Decomposition (SVD) to matrix v
    U, S, V_T = np.linalg.svd(v)
    
    # Extract the last column of V_T (transpose of V), which corresponds to the smallest singular value
    b = V_T.T[:, -1]

    # Rearrange b
    B = arrangeB(b)
    return B

def getA(B):
    """
    Extracts the intrinsic camera parameters from matrix B and returns the camera matrix A.
    
    @params:
        B (np.ndarray): Matrix B derived from the homography matrices
    @return:
        np.ndarray: The intrinsic camera matrix A, containing:
                    - alpha: focal length in terms of pixels along the x-axis
                    - beta: focal length in terms of pixels along the y-axis
                    - gamma: skew coefficient between x and y axis
                    - u0, v0: coordinates of the principal point (optical center)

    """
    # y-coordinate of the camera principal point
    v0 = (B[0,1]* B[0,2] - B[0,0]*B[1,2]) / (B[0,0]*B[1,1] - B[0,1]**2)

    # scale factor for the matrix
    lambd = (B[2,2] - (B[0,2]**2 + v0* (B[0,1]*B[0,2] - B[0,0]*B[1,2])) / B[0,0])

    # focal length along the x-axis
    alpha = np.sqrt(lambd / B[0,0])

    # focal length along the y-axis
    beta = np.sqrt((lambd* B[0,0]) / (B[0,0]*B[1,1] - B[0,1]**2))
    
    # skew coefficient between x and y axis
    gamma = -1* B[0,1]* (alpha**2)* (beta) / lambd
    
    # x-coordinate of the camera principal point
    u0 = (gamma*v0 /beta) - (B[0,2]* (alpha**2)/lambd)
    
    # camera matrix A
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    return A

def getRotAndTrans(A, H_all):
    """
    Calculate the rotation and translation matrices for each image from their corresponding homography matrices.
    
    @params:
        A (np.ndarray): Intrinsic camera matrix of shape (3, 3).
        H_all (np.ndarray): Array of homography matrices, each of shape (3, 3), one for each image.

    Returns:
        list: List of rotation and translation matrices for each image
    """
    # List to hold all rotation and translation matrices
    RT_all = []

    for H in H_all:
        # Decompose the homography matrix into the three columns
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]

        # Compute the scale factor lambda from the first column of H
        lambd = 1 / np.linalg.norm(np.matmul(np.linalg.pinv(A), h1), 2)
        r1 = np.matmul(lambd* np.linalg.pinv(A), h1)
        r2 = np.matmul(lambd* np.linalg.pinv(A), h2)
        r3 = np.cross(r1, r2)
        
        # Calculate the translation vector
        t = np.matmul(lambd*np.linalg.pinv(A),h3)
        
        # Assemble the rotation matrix and translation vector into a single matrix
        RT = np.vstack((r1, r2, r3, t)).T
        
        # Append the combined rotation and translation matrix to the list
        RT_all.append(RT)
    
    return RT_all

def extractParamFromA(A, K_distortion_init):
    """
    Extracts camera intrinsic parameters and initial distortion coefficients from the provided matrices
    
    @params:
        A (np.ndarray): The 3x3 intrinsic camera matrix
        K_distortion_init (np.ndarray): Array containing the initial estimates of the radial distortion coefficients

    @returns:
        np.ndarray: An array of extracted parameters [alpha, gamma, beta, u0, v0, k1, k2].
    """
    
    # Extract the intrinsic parameters from the camera matrix A
    alpha = A[0,0]
    gamma = A[0,1]
    u0 = A[0,2]
    beta = A[1,1]
    v0 = A[1,2]

    # Extract the distortion coefficients from the provided array
    k1 = K_distortion_init[0]
    k2 = K_distortion_init[1]

    return np.array([alpha, gamma, beta, u0, v0, k1, k2])

def retrieveA(x0):
    """
    Extracts the camera intrinsic matrix and distortion coefficients from a parameter vector

    @params:
        x0 (array-like): Vector containing camera parameters

    @returns:
        A (np.ndarray): Camera intrinsic matrix (focal lengths, skew, and principal point coordinates)
        K_distortion (np.ndarray): Radial distortion coefficients
    """
    alpha, gamma, beta, u0, v0, k1, k2 = x0

    # Construct the camera intrinsic matrix 'A'
    A = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])

    # Create the distortion coefficients array
    K_distortion = np.array([k1, k2])

    return A, K_distortion


def reprojectionRMSError(A, K_distortion, RT_all, images_corners, world_corners):
    """
    Calculates the reprojection error for all images in the dataset

    @params:
        A (np.ndarray): Intrinsic camera matrix
        K_distortion (np.ndarray): Distortion coefficients
        RT_all (list of np.ndarray): List of rotation and translation matrices for each image
        images_corners (list of np.ndarray): List of detected image corners for each image
        world_corners (np.ndarray): 3D world coordinates of the corners used during the calibration

    @returns:
        tuple: 
            - A numpy array of mean reprojection errors for each image
            - A list of reprojected corners for all images
    """
    # Extract camera parameters and distortion coefficients
    alpha, gamma, beta, u0, v0, k1, k2 = extractParamFromA(A, K_distortion)
    
    error_all_images = []
    reprojected_corners_all = []

    # Loop over each image
    for i in range(len(images_corners)):
        img_corners = images_corners[i]
        RT = RT_all[i] # Pose from which the image was taken
        P_matrix = np.dot(A, RT)  # Projection matrix
        error_per_img = 0
        reprojected_img_corners = []

        # Loop over each corner detected in the image
        for j in range(len(img_corners)):
            world_point_corners_nonHomo_2d = world_corners[j]
            world_point_3d_Homo = np.array(                         # [4,1]
                [
                    [world_point_corners_nonHomo_2d[0]],
                    [world_point_corners_nonHomo_2d[1]],
                    [0],
                    [1],
                ], dtype= float
            )

            img_corner_nonHomo = img_corners[j]
            img_corner_Homo = np.array(                            # [3,1]
                [
                    [img_corner_nonHomo[0]],
                    [img_corner_nonHomo[1]],
                    [1]
                ], dtype=float
            )

            # Project 3D world points to 2D using the camera projection matrix
            pixel_coords = np.matmul(P_matrix, world_point_3d_Homo)
            u = pixel_coords[0] / pixel_coords[2]
            v = pixel_coords[1]  / pixel_coords[2]

            image_coords = np.matmul(RT, world_point_3d_Homo)
            x_norm = image_coords[0] / image_coords[2]
            y_norm = image_coords[1] / image_coords[2]
            
            # Apply distortion coefficients
            r = np.sqrt(x_norm**2 + y_norm**2)

            u_hat = u + (u-u0)* (k1* r**2 + k2* (r**4))
            v_hat = v + (v-v0)* (k1* r**2 + k2* (r**4))

            img_corner_Homo_hat = np.array(
                [u_hat, 
                 v_hat,
                 [1]], dtype=float
            )

            reprojected_img_corners.append((img_corner_Homo_hat[0],
                                            img_corner_Homo_hat[1]))
            
            # Compute reprojection error
            error_per_corner = np.linalg.norm(
                (img_corner_Homo - img_corner_Homo_hat), 2
            )
            error_per_img = error_per_img + error_per_corner
        

        reprojected_corners_all.append(reprojected_img_corners)
        error_all_images.append(error_per_img / 54)
    
    return np.array(error_all_images), np.array(reprojected_corners_all)


def loss_func(x0, RT_all, images_corners, world_corners):
    """
    This function is designed to be used with scipy.optimize.least_squares

    @params:
        x0 (np.ndarray): Initial estimates of the camera parameters
        RT_all (np.ndarray): Rotation and translation matrices for each image
        images_corners (np.ndarray): Observed corner points in the images
        world_corners (np.ndarray): Corresponding corner points in the world coordinates

    @returns:
        np.ndarray: Reprojection errors for each image.
    """


    A, K_distortion = retrieveA(x0)
    error_all_images, _ = reprojectionRMSError(A=A,
                                               K_distortion=K_distortion,
                                               RT_all=RT_all,
                                               images_corners=images_corners,
                                               world_corners=world_corners)
    
    return np.array(error_all_images)

def saveCalibrationResults(output_folder, camera_name, intrinsic_params, distortion_params, mean_error_pre, mean_error_post):
    """
    Saves the calibration parameters in a JSON file.
    """
    file_path = os.path.join(output_folder, f"{camera_name}_intrinsic.json")
    timestamp = datetime.datetime.now()

    calibration_data = {
        "Camera_ID": camera_name,
        "intrinsic": {
            "alpha": intrinsic_params[0][0],
            "beta": intrinsic_params[1, 1],
            "gamma": intrinsic_params[0, 1],
            "u0": intrinsic_params[0, 2],
            "v0": intrinsic_params[1, 2],
        },
        "distortion": {
            "k1": distortion_params[0],
            "k2": distortion_params[1]
        },
        "meanError_preOpt": mean_error_pre,
        "meanError_postOpt": mean_error_post,
        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(file_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)

    print(f"- Calibration parameters saved in {file_path}")

def read_chessboard_dimensions(folder_path):
    """
    Read the chessboard dimensions from a JSON file.
    @params:
        json_path (str): Path to the folder containing the json with chessboard dimensions.
    @return:
        tuple: Tuple containing the dimensions of the chessboard (rows,, columns)
    """
    # Load the JSON file
    filename = parameters.JSON_CHESSBOARD
    json_path = os.path.join(folder_path, filename)

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        
    rows = data['chessboard_dimensions']['rows']
    columns = data['chessboard_dimensions']['columns']

    return (rows, columns)
