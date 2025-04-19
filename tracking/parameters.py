import numpy as np
TRAIN_PATH = 'tracking/runs/detect/train4/'
VIDEO_SUBDIR = 'video/'
COORDS_DIR = 'tracking/coordinates/'
START_SEC = 155
END_SEC = 166

# Folder containing training data
TRACKING_FOLDER = './tracking'

# Pairs of cameras to process (each pair will be used for triangulation)
CAM_PAIRS = [
        ["3", "1"],
        ["1", "2"],
        ["2", "4"],
        ["4", "12"],
        ["12", "7"],
        ["7", "8"],
        ["8", "6"],
        ["6", "5"],
        ["5", "13"],
        ["13", "3"]
    ]

# Court dimensions constants (in meters)
COURT_LENGTH = 18.0  # Standard volleyball court length
COURT_WIDTH = 9.0    # Standard volleyball court width
NET_HEIGHT = 2.24    # Height of the net at center
NET_WIDTH = 1.0      # Width of the net poles
