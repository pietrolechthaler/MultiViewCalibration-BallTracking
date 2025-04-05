import cv2
import json
import os

# List of cameras to process
CAMERA_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
BASE_IMAGE_PATH = 'annotation-images/out{}.jpg'
BASE_JSON_PATH = 'annotation-dist/out{}-ann.json'
OUTPUT_PATH = 'annotation-images/out{}-dist-annotated.jpg'

def draw_points_on_image(image_path, json_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: image not found {image_path}")
        return False
    
    # Load the JSON file
    try:
        with open(json_path) as f:
            points_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found {json_path}")
        return False
    
    # Draw each valid point
    for point_id, data in points_data.items():
        if data.get("status") == "ok":
            x = int(data["coordinates"]["x"])
            y = int(data["coordinates"]["y"])
            
            # Draw circle and text
            cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
            text = f"ID:{point_id} ({x},{y})"
            cv2.putText(image, text, (x + 20, y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Created: {output_path}")
    return True

# Process all cameras
for cam_id in CAMERA_IDS:
    img_path = BASE_IMAGE_PATH.format(cam_id)
    json_path = BASE_JSON_PATH.format(cam_id)
    out_path = OUTPUT_PATH.format(cam_id)
    
    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    draw_points_on_image(img_path, json_path, out_path)

print("Processing completed for all cameras")
