#!/bin/bash

# Camera IDs to process
CAMERA_IDS=(1 2 3 4 5 6 7 8 12 13)
SUCCESS=true

echo "Starting sequential camera processing..."
for camera_id in "${CAMERA_IDS[@]}"; do
    echo -n "Processing camera ${camera_id}..."
    if python tracking/detection.py "$camera_id"; then
        echo "OK"
    else
        echo "FAILED"
        SUCCESS=false
        # Continue to next camera instead of exiting
    fi
done

if $SUCCESS; then
    echo "All cameras processed successfully"
    exit 0
else
    echo "Processing completed with errors"
    exit 1
fi