#!/bin/bash
# Start MediaPipe Pose Service

SERVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR=".venv"

cd "$SERVICE_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Check if service is already running
if ./status.sh | grep -q "is running"; then
    echo "MediaPipe Pose Service is already running. Use ./stop.sh first if you want to restart."
    exit 1
fi

echo "Starting MediaPipe Pose Service..."
source "$VENV_DIR/bin/activate"
python app.py
