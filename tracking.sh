#!/bin/bash

# Script to run tracking Python scripts with error handling

echo "Starting tracking pipeline..."
echo "----------------------------"

# Function to run a script with error handling
run_script() {
    local script_name=$1
    echo "> Running $script_name..."
    
    if python "$script_name"; then
        echo "$script_name completed successfully"
        return 0
    else
        echo "ERROR: $script_name failed to execute properly"
        return 1
    fi
}

# Run scripts sequentially
run_script "tracking/triangulation.py" || exit 1
run_script "tracking/ball_tracking.py" || exit 1

echo "----------------------------"
echo "Tracking pipeline completed."
exit 0