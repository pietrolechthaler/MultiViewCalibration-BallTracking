# Parameters

VIDEO_PATHS = [
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

# VIDEO_PATHS = [
#     ['./video/board/out12F.mp4', (7,5)],
# ]

SKIP_FRAMES = 25
SQUARE_SIZE = 28 #mm

OUT = "-gen"
DISTORTION = "-distortion"
JSON_CHESSBOARD = "chessboard.json"
SRC_GEN = './src-gen'
SETUP = "./setup.json"
LANDMARKS = "landmarks.json"
SAVE_UNDISTORTED = True

WORLD_LABEL_POINTS = {
    '1': [0, 9, 0],
    '2': [0, 0, 0],
    '3': [6, 0, 0],
    '4': [9, 0, 0],
    '5': [12, 0, 0],
    '6': [18, 0, 0],
    '7': [18, 9, 0],
    '8': [12, 9, 0],
    '9': [9, 9, 0],
    '10': [6, 9, 0],
    '11': [0, 6.9, 0],
    '12': [0, 2.1, 0],
    '13': [18, 2.1, 0],
    '14': [18, 6.9, 0]
}

CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]