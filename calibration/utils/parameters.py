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

SKIP_FRAMES = 10
SQUARE_SIZE = 28 #mm

OUT = "-gen"
DISTORTION = "-distortion"
JSON_CHESSBOARD = "chessboard.json"
SRC_GEN = './src-gen'
SETUP = "./setup.json"
LANDMARKS = "landmarks.json"
SAVE_UNDISTORTED = True

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
}

CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]