#!/bin/bash
# Check MediaPipe Pose Service status

SERVICE_NAME="python app.py"

if ps aux | grep "$SERVICE_NAME" | grep -v grep > /dev/null; then
    PIDS=$(ps aux | grep "$SERVICE_NAME" | grep -v grep | awk '{print $2}' | tr '\n' ' ')
    echo "MediaPipe Pose Service is running (PIDs: $PIDS)"
else
    echo "MediaPipe Pose Service is not running"
fi
