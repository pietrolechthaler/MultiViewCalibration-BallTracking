#!/bin/bash

# Close any existing instances of the server running on port 5000
fuser -k 5000/tcp

# Run the viewer
python viewer/app.py &

# Open the default web browser
xdg-open http://127.0.0.1:5000/
