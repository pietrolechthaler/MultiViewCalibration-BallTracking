import numpy as np
import cv2
import pickle
import os
import json
import argparse
from calibration.utils.parameters import WORLD_LABEL_POINTS, SRC_GEN

def load_calibration_data(camera_id):
    """Load intrinsic and extrinsic parameters for a given camera."""
    intrinsic_path = os.path.join(SRC_GEN, f'out{camera_id}F-gen', 'calibration_data.pkl')
    extrinsic_path = os.path.join(SRC_GEN, f'out{camera_id}F-gen', 'calibration_extrinsic.pkl')
    
    # Carica e filtra i parametri intrinseci
    with open(intrinsic_path, 'rb') as f:
        full_intrinsic = pickle.load(f)
        intrinsic = {
            'camera_matrix': full_intrinsic.get('camera_matrix'),
            'distortion_coefficients': full_intrinsic.get('distortion_coefficients')
        }
    
    # Carica e filtra i parametri extrinseci
    with open(extrinsic_path, 'rb') as f:
        full_extrinsic = pickle.load(f)
        extrinsic = {
            'rotation_matrix': full_extrinsic.get('rotation_matrix'),
            'translation_vector': full_extrinsic.get('translation_vector')
        }
    
    return intrinsic, extrinsic

def load_annotations(camera_id):
    """Load world coordinates of key points from annotations."""
    with open(os.path.join('annotation-dist', f'out{camera_id}-ann.json'), 'r') as f:
        annotations = json.load(f)
    return annotations

def find_homography(id1, id2):
    """Find the homography matrix between two cameras using common world points."""
    ann1 = load_annotations(id1)
    ann2 = load_annotations(id2)

    if not ann1 or not ann2:
        return None

    src_pts, dst_pts = [], []

    for point_id in ann1:
        if point_id in ann2 and "coordinates" in ann1[point_id] and "coordinates" in ann2[point_id]:
            src_pts.append([ann1[point_id]["coordinates"]["x"], ann1[point_id]["coordinates"]["y"]])
            dst_pts.append([ann2[point_id]["coordinates"]["x"], ann2[point_id]["coordinates"]["y"]])

    if len(src_pts) < 4:
        print("Not enough common points to compute homography.")
        return None

    #print(f"Source points: {src_pts}")
    #print(f"Destination points: {dst_pts}")
    # Convert to NumPy array with correct shape (N,1,2)
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)

    #print(f"Source points: {src_pts}")
    #print(f"Destination points: {dst_pts}")
    
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RHO) # sotituito cv2.RANSAC con cv2.RHO
    return H

def find_corresponding_point(id1, x1, y1, id2):
    """Find the corresponding point in camera `id2` for a given point in camera `id1`."""
    intrinsic1, extrinsic1 = load_calibration_data(id1)
    #print(f"Intrinsic1: {intrinsic1}")
    #print(f"Extrinsic1: {extrinsic1}")
    intrinsic2, extrinsic2 = load_calibration_data(id2)
    #print(f"Intrinsic2: {intrinsic2}")
    #print(f"Extrinsic2: {extrinsic2}")  

    if not all([intrinsic1, extrinsic1, intrinsic2, extrinsic2]):
        return None

    H = find_homography(id1, id2)
    
    if H is None:
        return None
    
    print(f"Homography matrix: {H}")
    print(np.linalg.det(H))  # Un valore troppo piccolo potrebbe indicare problemi: es. 1 e 4, solo 4 punti in comune, aggiungere più vicini al centro campo
    print(H[2,2])  # Dovrebbe essere vicino a 1


    mtx1, dist1 = intrinsic1['camera_matrix'], intrinsic1['distortion_coefficients']

    # Convert (x1, y1) to NumPy array and reshape
    src_pt = np.array([[x1, y1]], dtype=np.float32).reshape(-1, 1, 2)

    # Undistort the source point
    undistorted_pts = cv2.undistortPoints(src_pt, mtx1, dist1, None, mtx1)

    # Reshape to match (N,1,2) before applying perspective transform
    undistorted_pts = undistorted_pts.reshape(1, 1, 2)  # Ensure correct shape

    # Apply homography transformation
    transformed_pts = cv2.perspectiveTransform(undistorted_pts, H)
    x2, y2 = transformed_pts[0][0]  # Extract coordinates

    return x2, y2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find corresponding point between two cameras.')
    parser.add_argument('id1', type=int, help='Source camera ID')
    parser.add_argument('x1', type=float, help='X coordinate in source camera')
    parser.add_argument('y1', type=float, help='Y coordinate in source camera')
    parser.add_argument('id2', type=int, help='Target camera ID')
    
    args = parser.parse_args()
    
    if args.id1 == args.id2:
        print(f"Corresponding coordinates in camera {args.id2}: ({args.x1:.2f}, {args.y1:.2f})")
    else:
        result = find_corresponding_point(args.id1, args.x1, args.y1, args.id2)
        if result:
            x2, y2 = result
            print(f"Corresponding coordinates in camera {args.id2}: ({x2:.2f}, {y2:.2f})")
        else:
            print("Failed to find corresponding point.")
