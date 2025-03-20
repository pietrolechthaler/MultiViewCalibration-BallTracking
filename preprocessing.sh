#!/bin/bash

# Define the output folder
ARTIFACTS="./src-gen"

# Check if the output folder exists and remove it
if [ -d "$ARTIFACTS" ]; then
    echo "----------------------------------------"
    echo "> Removing existing artifacts folder"
    rm -rf "$ARTIFACTS"
fi

# Run the Python script to extract frames from videos
echo "----------------------------------------"
echo "> Extracting frames from videos"
python3 calibration/utils/preprocessing.py