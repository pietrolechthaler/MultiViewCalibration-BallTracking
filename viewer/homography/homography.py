import numpy as np
import cv2
import os
import json

WORLD_LABEL_POINTS = {
    '1': [-9, -4.5, 0],
    '2': [-9, 4.5, 0],
    '3': [-3, 4.5, 0],
    '4': [0, 4.5, 0],
    '5': [3, 4.5, 0],
    '6': [9, 4.5, 0],
    '7': [9, -4.5, 0],
    '8': [3, -4.5, 0],
    '9': [0, -4.5, 0],
    '10': [-3, -4.5, 0],
    '11': [-9, -2.45, 0],
    '12': [-9, 2.45, 0],
    '13': [9, 2.45, 0],
    '14': [9, -2.45, 0],
    '15': [-14, -7.5, 0],
    '16': [-14, 7.5, 0],
    '17': [14, 7.5, 0],
    '18': [14, -7.5, 0],
    '19': [-14, -2.45, 0],
    '20': [-14, 2.45, 0],
    '21': [14, 2.45, 0],
    '22': [14, -2.45, 0],
    '23': [-8.2, -2.45, 0],
    '24': [-8.2, -1.75, 0],
    '25': [-8.2, 1.75, 0],
    '26': [-8.2, 2.45, 0],
    '27': [8.2, 2.45, 0],
    '28': [8.2, 1.75, 0],
    '29': [8.2, -1.75, 0],
    '30': [8.2, -2.45, 0],
    '31': [0, -7.5, 0],
    '32': [0, -1.75, 0],
    '33': [0, 1.75, 0],
    '34': [0, 7.5, 0],
    '35': [-3, 0, 0],
    '36': [3, 0, 0]
}

CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]

SKETCH_LABEL_POINTS = {
    '1': [48, 183.125, 0],
    '2': [48, 44.125, 0],
    '3': [141.667, 44.125, 0],
    '4': [188.5, 44.125, 0],
    '5': [235.333, 44.125, 0],
    '6': [329, 44.125, 0],
    '7': [329, 183.125, 0],
    '8': [235.333, 183.125, 0],
    '9': [188.5, 183.125, 0],
    '10': [141.667, 183.125, 0],
    '11': [48, 151.464, 0],
    '12': [48, 75.786, 0],
    '13': [329, 75.786, 0],
    '14': [329, 151.464, 0],
    '15': [-30.056, 229.458, 0],
    '16': [-30.056, -2.208, 0],
    '17': [407.056, -2.208, 0],
    '18': [407.056, 229.458, 0],
    '19': [-30.056, 151.464, 0],
    '20': [-30.056, 75.786, 0],
    '21': [407.056, 75.786, 0],
    '22': [407.056, 151.464, 0],
    '23': [60.489, 151.464, 0],
    '24': [60.489, 140.653, 0],
    '25': [60.489, 86.597, 0],
    '26': [60.489, 75.786, 0],
    '27': [316.511, 75.786, 0],
    '28': [316.511, 86.597, 0],
    '29': [316.511, 140.653, 0],
    '30': [316.511, 151.464, 0],
    '31': [188.5, 229.458, 0],
    '32': [188.5, 140.653, 0],
    '33': [188.5, 86.597, 0],
    '34': [188.5, -2.208, 0],
    '35': [141.667, 113.625, 0],
    '36': [235.333, 113.625, 0]
}

def get_world_point(x_stylized, y_stylized):
    """
    Converts a point from the stylized field map image (in pixels)
    to the real world using the mapping defined by correspondences.
    Only x and y components are considered (ignoring the z component).
    """
    # Known points in the field map (in pixels)
    src_pts = np.array([[SKETCH_LABEL_POINTS[str(i)][0], SKETCH_LABEL_POINTS[str(i)][1]] 
                    for i in range(1,37)], dtype=np.float32).reshape(-1, 1, 2)
    
    # Corresponding real-world coordinates (extracted from WORLD_LABEL_POINTS)
    dst_pts = np.array([[WORLD_LABEL_POINTS[str(i)][0], WORLD_LABEL_POINTS[str(i)][1]] 
                    for i in range(1,37)], dtype=np.float32).reshape(-1, 1, 2)
    
    # Calculate homography from field map to world coordinate system
    H_field, _ = cv2.findHomography(src_pts, dst_pts, method=0)
    input_pt = np.array([[x_stylized, y_stylized]], dtype=np.float32).reshape(-1, 1, 2)
    world_pt = cv2.perspectiveTransform(input_pt, H_field)
    return world_pt[0][0]  # Returns (x, y) in the real world

def load_annotations(camera_id):
    """
    Loads the JSON file containing annotations for the camera.
    Annotations must contain key points with view coordinates.
    """
    ann_file = os.path.join('./annotation/annotation-dist', f'out{camera_id}-ann.json')
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def find_homography_field(camera_id):
    """
    Calculates the homography matrix that transforms points from the real world 
    (defined by WORLD_LABEL_POINTS) to the camera view. 
    """
    ann = load_annotations(camera_id)
    src_pts = []  # Points from WORLD_LABEL_POINTS
    dst_pts = []  # Points from annotations
    
    for point_id in ann:
        if point_id in ann and "coordinates" in ann[point_id]:
            
            # Add world coordinates (X,Y)
            src_pts.append([
                WORLD_LABEL_POINTS[point_id][0], 
                WORLD_LABEL_POINTS[point_id][1]
            ])
            
            # Add image coordinates (x,y)
            dst_pts.append([
                ann[point_id]["coordinates"]["x"],
                ann[point_id]["coordinates"]["y"]
            ])

    #print(f"Camera {camera_id}: {len(src_pts)} common points found.")

    if len(src_pts) < 4:
        print(f"Not enough common points to calculate homography for camera out{camera_id}.")
        return None
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 1, 2)
    
    # Calculate homography that transforms the real world to the camera view
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return H

def get_camera_point(camera_id, x_world, y_world):
    """
    Transforms point (x_world, y_world) from the real world to the camera view.
    into coordinates of the camera view identified by camera_id.
    """
    H = find_homography_field(camera_id)
    if H is None:
        return None
    src_pt = np.array([[x_world, y_world]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_pts = cv2.perspectiveTransform(src_pt, H)
    x_cam, y_cam = transformed_pts[0][0]
    return x_cam, y_cam

def getCorrespondences(x, y):

    # Convert input point from stylized field map to real world
    x_world, y_world = get_world_point(x, y)
    print(f"- Coordinates of real world: ({x_world:.3f}, {y_world:.3f})")

    results = {}  # Using regular dict instead of OrderedDict

    for cam_id in CAMERA_IDS:
        pt = get_camera_point(cam_id, x_world, y_world)
        if pt is None:
            print(f"Transformation failed for camera out{cam_id}.")
        else:
            # Convert numpy floats to native Python floats and round
            x_cam = float(round(pt[0], 3))
            y_cam = float(round(pt[1], 3))
            results[f"out{cam_id}"] = {"x": x_cam, "y": y_cam}
    
    return results
    
if __name__ == "__main__":
    getCorrespondences(100,200)