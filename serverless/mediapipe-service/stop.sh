#!/bin/bash
# Stop MediaPipe Pose Service

SERVICE_NAME="python app.py"
PID=$(ps aux | grep "$SERVICE_NAME" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "MediaPipe Pose Service is not running"
else
    echo "Stopping MediaPipe Pose Service (PID: $PID)..."
    kill "$PID"
    echo "Service stopped"
fi
