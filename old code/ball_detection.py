import cv2
import numpy as np

def cumulative_frame_difference(video_path, output_image_path, max_duration=50, start_time=42, threshold=30, S_max=255):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {fps}")
    
 
    start_frame = int(fps * start_time)
    max_frames = int(fps * max_duration)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, previous_frame = cap.read()
    
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    cumulative_diff = np.zeros_like(previous_frame, dtype=np.float32)
    
    frame_count = 0
    
    while True:
        print(f"Frame: {start_frame} of {max_frames - start_frame}")
        ret, current_frame = cap.read()
        if not ret or frame_count >= max_frames - start_frame:
            break
    
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        frame_diff = cv2.absdiff(current_frame, previous_frame)
        
        cumulative_diff = np.where(frame_diff > threshold, S_max, np.maximum(cumulative_diff - 1, 0))
        S_t_uint8 = cumulative_diff.astype(np.uint8)
        
        kernel_o = np.ones((10, 10), np.uint8)
        kernel_c = np.ones((12, 12), np.uint8)
        S_t_cleaned = cv2.morphologyEx(S_t_uint8, cv2.MORPH_CLOSE, kernel_c)
        S_t_cleaned = cv2.morphologyEx(S_t_cleaned, cv2.MORPH_OPEN, kernel_o)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(S_t_cleaned, connectivity=8)
        
        ball_candidates = np.zeros_like(S_t_cleaned)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            aspect_ratio = width / height

            if 500 < area < 5000 and 0.5 < aspect_ratio < 2.0:
                ball_candidates[labels == i] = 255
        

        previous_frame = current_frame
        
        #cv2.imshow('Frame', ball_candidates)
        #cv2.waitKey(1)
        
        frame_count += 1

    cumulative_diff = np.clip(cumulative_diff, 0, 255).astype(np.uint8)
    S_t_cleaned = np.clip(S_t_cleaned, 0, 255).astype(np.uint8)
    
    cv2.imwrite(output_image_path, cumulative_diff)
    cv2.imwrite('S_t_cleaned.png', S_t_cleaned)
    cv2.imwrite('ball_candidates.png', ball_candidates)

    cap.release()
    print("Ball detection completed.")

video_path = '../out1.mp4'
output_image_path = 'cumulative_frame_difference_t30.png'
cumulative_frame_difference(video_path, output_image_path)
