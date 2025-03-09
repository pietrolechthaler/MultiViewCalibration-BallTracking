#!/bin/bash

# Base path for input and output folders
base_input_folder="src-gen"
output_folder="video/output_corrected"

# Create the output folder if it doesn't exist
mkdir -p "$output_folder"

# Iterate from out1F to out13F
for i in {1..13}; do
    # Build the folder name (e.g., out1F, out2F, ..., out13F)
    folder_name="out${i}F"
    input_folder="${base_input_folder}/${folder_name}"

    # Check if the input folder exists
    if [ -d "$input_folder" ]; then
        echo "Processing folder: $input_folder"

        # Path to the calibration JSON file
        calibration_json="${input_folder}-gen/${folder_name}_intrinsic.json"

        # Path to the input video
        input_video="video/board/${folder_name}.mp4"    

        # Path to the output video
        output_video="${output_folder}/${folder_name}-calib_video.mp4"

        # Run the Python script to correct the video
        python calibration/undistortion.py "$calibration_json" "$input_video" "$output_video"

    else
        echo "Folder not found: $input_folder"
    fi
done

echo "Processing completed!"