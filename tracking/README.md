Add here all scripts related to ball tracking

- ball_detection.py: 
    cumulatve frame difference is computed for every pixel
    threshold M
    morphological operations @every frame 
    object filtered depending on area sizes and shape
    NB: brightness of ball candidates of past frame is gradually decreased every frame