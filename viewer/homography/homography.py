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

import numpy as np

def get_world_point(x, y):
    world_corners = np.array([
        [-9, -4.5, 0],
        [-9, 4.5, 0],
        [9, 4.5, 0],
        [9, -4.5, 0]
    ])

    court_corners = np.array([
        [63, 241.125, 0],
        [63, 57.125, 0],
        [438, 57.125, 0],
        [438, 241.125, 0],
    ])

    # Check if point is inside the court
    if (x < court_corners[0, 0] or x > court_corners[2, 0] or y < court_corners[1, 1] or y > court_corners[0, 1]):
        return None,None  #outside

    # Calculate the width and height of the court and world
    court_width = court_corners[2, 0] - court_corners[0, 0]
    court_height = court_corners[0, 1] - court_corners[1, 1]
    
    world_width = world_corners[2, 0] - world_corners[0, 0]
    world_height = world_corners[0, 1] - world_corners[1, 1]

    # Calculate the world coordinates
    world_x = ((x - court_corners[0, 0]) / court_width) * world_width + world_corners[0, 0]
    world_y = ((y - court_corners[1, 1]) / court_height) * world_height + world_corners[1, 1]

    return world_x, world_y


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

    print(f"Camera {camera_id}: {len(src_pts)} common points found.")

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

def getCorrespondences(x_world, y_world):

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