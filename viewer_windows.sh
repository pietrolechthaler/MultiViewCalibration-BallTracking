#!/bin/bash

# Close any existing Flask server running on port 5000
cmd "/C taskkill /F /IM python.exe /T" > /dev/null 2>&1

# Run the Flask app in the background
python viewer/app.py &

# Open the default web browser
start "" "http://127.0.0.1:5000/"
