#!/bin/bash

# Configuration
input_dir="video/match"
output_dir="src-gen/landsmark"
calib_root="src-gen"

# Create output directory
mkdir -p "$output_dir"

# extract the first frame from the video using calibration/extract_first_frame.py
python calibration/utils/extract_first_frame.py "$input_dir" "$output_dir"

# Process images 1-3
for i in {1,3}; do
    calib_folder="${calib_root}/out${i}F-gen"
    input_img="${output_dir}/out${i}.jpg" 
    output_img="${output_dir}/out${i}-undist.jpg"

    
    if [ -d "$calib_folder" ] && [ -f "$input_img" ]; then
        echo "Processing out${i} with calibration from $calib_folder"
        python calibration/undistortion.py "$calib_folder" "$input_img" "$output_img"
    else
        echo "Skipping out${i} - missing calibration or input image"
    fi
done

echo "Undistortion process completed"