#!/bin/bash

# Define the path to the 'src-gen' directory
target_directory="src-gen/landsmark"

# Create 'landsmark' directory inside 'src-gen' if it doesn't exist
mkdir -p "$target_directory"

# Navigate to the video directory
cd "video/match" || exit

# Loop through all mp4 files and extract the first frame
for video in *.mp4; do
    # Define the output filename (change extension to .jpg)
    output_file="${video%.mp4}.jpg"
    
    # Use FFmpeg to extract the first frame and save it to the 'landsmark' directory
    ffmpeg -i "$video" -frames:v 1 "../../$target_directory/$output_file"
done

echo "Extraction complete."
