#!/bin/bash
# Check MediaPipe Pose Service status

SERVICE_NAME="python app.py"

if ps aux | grep "$SERVICE_NAME" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "$SERVICE_NAME" | grep -v grep | awk '{print $2}')
    echo "MediaPipe Pose Service is running (PID: $PID)"
else
    echo "MediaPipe Pose Service is not running"
fi
