import cv2
import numpy as np
import csv
import pickle
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path configurations
SRC_GEN_PATH = '../src-gen'  # Path to generated source files
TRAIN_FOLDER = 'runs/detect/train4'  # Folder containing detection results
VIDEO_PATH = '../video/match'  # Path to video files

# Volleyball court dimensions (in meters)
COURT_LENGTH = 18.0  
COURT_WIDTH = 9.0    
NET_HEIGHT = 2.24    # Official net height for women's volleyball
NET_WIDTH = 1.0      # Net width visualization parameter
LINE_WIDTH = 0.05    # Line width for court markings

def get_video_dimensions(cam_id):
    """Gets width and height from the video corresponding to the specified camera"""
    video_path = os.path.join(VIDEO_PATH, f'out{cam_id}.mp4')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width in pixels
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height in pixels
    cap.release()
    return width, height

def load_detections(csv_path, cam_id):
    """
    Loads detections from CSV file maintaining original coordinates
    Returns dictionary with (frame_idx, ball_idx) as keys and (x, y, confidence) as values
    """
    frame_width, frame_height = get_video_dimensions(cam_id)
    
    detections = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 5:
                continue
            frame_idx, ball_idx, x, y, conf = row
            key = (int(frame_idx), int(ball_idx))
            detections[key] = (float(x), float(y), float(conf))
    return detections, frame_width, frame_height

def load_calibration(cam_id, src_gen_path):
    """Loads intrinsic and extrinsic calibration parameters"""
    cam_folder = os.path.join(src_gen_path, f'out{cam_id}F-gen')
    camera_matrix = np.loadtxt(os.path.join(cam_folder, 'camera_matrix.txt'), dtype=np.float32)  # Intrinsic parameters
    
    with open(os.path.join(cam_folder, 'calibration_extrinsic.pkl'), 'rb') as f:
        extrinsic_data = pickle.load(f)
    rotation_matrix = extrinsic_data['rotation_matrix']  # Camera rotation
    translation_vector = extrinsic_data['translation_vector']  # Camera translation
    
    P = get_projection_matrix(camera_matrix, rotation_matrix, translation_vector)
    return P, camera_matrix, rotation_matrix, translation_vector

def get_projection_matrix(K, R, t):
    """Constructs the projection matrix in the correct format"""
    RT = np.hstack((R, t.reshape(3,1)))  # Combined rotation-translation matrix
    P = K @ RT  # Projection matrix = intrinsic @ extrinsic
    return P

def triangulate_points(P1, P2, point1, point2):
    """
    Triangulate a 3D point from two different pov but correspondent 2D points 
    """

    # Convert points to correct format for OpenCV
    point1 = np.array([[point1[0], point1[1]]], dtype=np.float32)
    point2 = np.array([[point2[0], point2[1]]], dtype=np.float32)
    
    # Triangulate points (using Direct Linear Transform)
    point4d = cv2.triangulatePoints(P1, P2, point1.T, point2.T)
    
    # Convert from homogeneous
    point3d = cv2.convertPointsFromHomogeneous(point4d.T)[0][0]
        
    return point3d 

def draw_volleyball_court(ax):
    """Draws a 3D volleyball court visualization"""
    hl = COURT_LENGTH/2  # Half length
    hw = COURT_WIDTH/2   # Half width
    
    # Boundary lines
    ax.plot([-hl, hl], [-hw, -hw], [0, 0], 'white', linewidth=2)
    ax.plot([-hl, hl], [hw, hw], [0, 0], 'white', linewidth=2)
    ax.plot([-hl, -hl], [-hw, hw], [0, 0], 'white', linewidth=2)
    ax.plot([hl, hl], [-hw, hw], [0, 0], 'white', linewidth=2)
    
    # Center lines
    ax.plot([-hl, hl], [0, 0], [0, 0], 'white', linestyle='--')
    
    # Attack zones
    attack_line = 3.0  # 3-meter line from center
    ax.plot([+attack_line, +attack_line], [-hw, hw], [0, 0], 'white')
    ax.plot([-attack_line, -attack_line], [-hw, hw], [0, 0], 'white')
    
    # Net visualization
    ax.plot([-NET_WIDTH/2, NET_WIDTH/2], [0, 0], [NET_HEIGHT, NET_HEIGHT], 'red', linewidth=2)
    ax.plot([0, 0], [-hw, hw], [NET_HEIGHT, NET_HEIGHT], 'red', linestyle=':')
    
    # Net poles
    ax.plot([0, 0], [-hw, -hw], [0, NET_HEIGHT], 'red', linewidth=1)
    ax.plot([0, 0], [hw, hw], [0, NET_HEIGHT], 'red', linewidth=1)

def plot_world_points(world_points, cam1_id, cam2_id):
    """Visualizes 3D points with volleyball court reference"""
    if not world_points:
        print("No valid points to display")
        return
    
    world_points = np.array([p for p in world_points if p is not None])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    draw_volleyball_court(ax)
    
    ax.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2],
               c='yellow', marker='o', s=30, label='Ball')
    
    ax.set_xlabel('X (court length) [m]')
    ax.set_ylabel('Y (court width) [m]')
    ax.set_zlabel('Z (height) [m]')
    
    ax.set_xlim(-COURT_LENGTH/2 - 1, COURT_LENGTH/2 + 1)
    ax.set_ylim(-COURT_WIDTH/2 - 1, COURT_WIDTH/2 + 1)
    ax.set_zlim(0, 5)
    
    ax.set_title('Ball positions')
    ax.legend()
    plt.tight_layout()
    
    plot_filename = os.path.join(TRAIN_FOLDER, f'world_positions_cam{cam1_id}_cam{cam2_id}.png')
    fig.savefig(plot_filename, dpi=300)
    plt.show()

def main():
    # CSV file paths
    csv_cam1 = os.path.join(TRAIN_FOLDER, 'coordinates', 'out1_coordinates.csv')
    csv_cam2 = os.path.join(TRAIN_FOLDER, 'coordinates', 'out6_coordinates.csv')
    
    # Load detections from both cameras
    cam1_id, cam2_id = 1, 6
    detections_cam1, width1, height1 = load_detections(csv_cam1, cam1_id)
    detections_cam2, width2, height2 = load_detections(csv_cam2, cam2_id)
    
    print(f"Frame dimensions Camera {cam1_id}: {width1}x{height1}")
    print(f"Frame dimensions Camera {cam2_id}: {width2}x{height2}")
    
    # Load camera calibration parameters
    P1, K1, R1, t1 = load_calibration(cam1_id, SRC_GEN_PATH)
    P2, K2, R2, t2 = load_calibration(cam2_id, SRC_GEN_PATH)
    
    # Verify projection matrices
    print("\nProjection matrix Camera 1:\n", P1)
    print("\nProjection matrix Camera 2:\n", P2)
    
    # Find common detections with minimum confidence (ora come ora inutile: risultati da train già filtrati per conf > 0.4, da valutare se ulteriore filtro)
    common_keys = set(detections_cam1.keys()).intersection(set(detections_cam2.keys()))
    min_confidence = 0.4  # Threshold for detection reliability
    common_keys = [k for k in common_keys 
                  if detections_cam1[k][2] > min_confidence 
                  and detections_cam2[k][2] > min_confidence]
    
    print(f"\nFound {len(common_keys)} common detections with confidence > {min_confidence}")
    
    # Perform triangulation
    world_points = []
    output_filename = os.path.join(TRAIN_FOLDER, f"world_positions_cam{cam1_id}_cam{cam2_id}.csv")
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_idx', 'ball_idx', 'world_x', 'world_y', 'world_z'])
        
        valid_points = 0
        for key in common_keys:
            frame_idx, ball_idx = key
            pt1 = detections_cam1[key][:2] 
            pt2 = detections_cam2[key][:2] 
            
            point_3d = triangulate_points(P1, P2, pt1, pt2)
            
            if point_3d is not None:
                # Define valid coordinates: within court boundaries
                world_points.append(point_3d)
                writer.writerow([frame_idx, ball_idx, *point_3d])                
                if abs(point_3d[0]) < COURT_LENGTH/2 and abs(point_3d[1]) < COURT_WIDTH/2:
                    # world_points.append(point_3d) # da inserire qua una volta funzionante 
                    # writer.writerow([frame_idx, ball_idx, *point_3d])
                    valid_points += 1
                else:
                    print(f"Triangulation failed for frame {frame_idx}, ball {ball_idx}: out of bounds")
            else:
                print(f"Triangulation failed for frame {frame_idx}, ball {ball_idx}")
    
    print(f"\nValid 3D points calculated: {valid_points}/{len(common_keys)}")
    
    # Visualization
    if world_points:
        plot_world_points(world_points, cam1_id, cam2_id)
    else:
        print("No valid 3D points to plot.")

if __name__ == "__main__":
    main()