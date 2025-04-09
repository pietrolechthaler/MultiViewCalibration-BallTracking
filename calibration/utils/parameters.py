import numpy as np

# Video paths and associated chessboard dimensions
CALIBRATION_VIDEO = [
    ['./video/board/out1F.mp4', (7,5)],
    ['./video/board/out2F.mp4', (7,5)],
    ['./video/board/out3F.mp4', (7,5)],
    ['./video/board/out4F.mp4', (7,5)],
    ['./video/board/out5F.mp4', (9,6)],
    ['./video/board/out6F.mp4', (9,6)],
    ['./video/board/out7F.mp4', (7,5)],
    ['./video/board/out8F.mp4', (9,6)],
    ['./video/board/out12F.mp4', (7,5)],
    ['./video/board/out13F.mp4', (7,5)],
]

# Number of frames to skip during processing
SKIP_FRAMES = 25

# Size of the chessboard square in millimeters
SQUARE_SIZE = 28

# Maximum number of images to consider in ranking procedure
TOP_N = 40

# Suffix for generated files
OUT = "-gen"
DISTORTION = "-distortion"

# Name of the JSON file containing chessboard dimensions
JSON_CHESSBOARD = "chessboard.json"

# Path to the folder for generated files
SRC_GEN = './src-gen'

# Flag to save undistorted images
SAVE_UNDISTORTED = False

# World reference points for calibration
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

# Real-world camera positions
REAL_WORLD_CAMERA_POSITIONS = [
        np.array([-14.5, -17.7, 6.2]),  # out1
        np.array([0.0, -17.7, 6.2]),    # out2
        np.array([-22.0, -10.0, 6.6]),  # out3
        np.array([14.5, -17.7, 6.2]),   # out4
        np.array([-22.0, 10.0, 5.8]),   # out5
        np.array([0.0, 10.0, 6.3]),     # out6
        np.array([25.0, 0.0, 6.4]),     # out7
        np.array([22.0, 10.0, 6.3]),    # out8
        np.array([22.0, -10.0, 6.9]),   # out12
        np.array([-22.0, 0.0, 7]),      # out13
    ]

# Camera identifiers
CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]