#!/bin/bash
# Stop MediaPipe Pose Service

SERVICE_NAME="python app.py"
PIDS=$(ps aux | grep "$SERVICE_NAME" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "MediaPipe Pose Service is not running"
else
    echo "Stopping MediaPipe Pose Service (PIDs: $PIDS)..."
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    # Check if any processes are still running
    REMAINING=$(ps aux | grep "$SERVICE_NAME" | grep -v grep | wc -l)
    if [ "$REMAINING" -gt 0 ]; then
        echo "Force stopping remaining processes..."
        for pid in $PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    echo "Service stopped"
fi
