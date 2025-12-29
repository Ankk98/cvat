#!/usr/bin/env python3
"""
Test MediaPipe coordinate system and verify coordinate transformation.
"""

import base64
import requests
import json
import cv2
from PIL import Image
import io

def test_coordinates():
    # Load test image
    img_path = '../test-scripts/test-data/egocentric-hands/001.jpg'
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h} (width x height)")

    # Encode image
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()

    # Call MediaPipe service
    resp = requests.post('http://localhost:8000/detect',
                        json={'image': img_b64, 'threshold': 0.3},
                        timeout=30)
    data = json.loads(resp.text)

    if not data:
        print("No detections returned")
        return

    skel = data[0]
    elements = skel.get('elements', [])

    print(f"\nSkeleton elements: {len(elements)}")
    print("\nCoordinate analysis:")
    print("=" * 80)

    # Analyze coordinates
    hand_keypoints = []
    pose_keypoints = []

    for elem in elements:
        pts = elem.get('points', [])
        if len(pts) >= 2:
            x, y = pts[0], pts[1]
            label = elem.get('label', 'unknown')

            # Check if coordinates are reasonable
            x_norm = x / w if w > 0 else 0
            y_norm = y / h if h > 0 else 0

            if 'hand' in label.lower() or 'wrist' in label.lower() or 'thumb' in label.lower() or 'finger' in label.lower():
                hand_keypoints.append((label, x, y, x_norm, y_norm))
            else:
                pose_keypoints.append((label, x, y, x_norm, y_norm))

    print("\nHand Keypoints (first 10):")
    for label, x, y, x_norm, y_norm in hand_keypoints[:10]:
        status = ""
        if x > w * 0.8 and y > h * 0.8:
            status = " [BOTTOM RIGHT!]"
        elif x < 0 or x > w or y < 0 or y > h:
            status = " [OUT OF BOUNDS!]"
        print(f"  {label:25s} -> ({x:6.1f}, {y:6.1f}) | Normalized: ({x_norm:.3f}, {y_norm:.3f}){status}")

    print("\nPose Keypoints (first 10):")
    for label, x, y, x_norm, y_norm in pose_keypoints[:10]:
        status = ""
        if x > w * 0.8 and y > h * 0.8:
            status = " [BOTTOM RIGHT!]"
        elif x < 0 or x > w or y < 0 or y > h:
            status = " [OUT OF BOUNDS!]"
        print(f"  {label:25s} -> ({x:6.1f}, {y:6.1f}) | Normalized: ({x_norm:.3f}, {y_norm:.3f}){status}")

    # Check for problematic coordinates
    bottom_right_count = sum(1 for _, x, y, _, _ in hand_keypoints + pose_keypoints
                            if x > w * 0.8 and y > h * 0.8)

    if bottom_right_count > 0:
        print(f"\n⚠️  WARNING: {bottom_right_count} keypoints are in bottom-right corner!")
        print("   This suggests a coordinate transformation issue.")

    # Check coordinate ranges
    all_x = [x for _, x, _, _, _ in hand_keypoints + pose_keypoints]
    all_y = [y for _, _, y, _, _ in hand_keypoints + pose_keypoints]

    if all_x and all_y:
        print(f"\nCoordinate ranges:")
        print(f"  X: min={min(all_x):.1f}, max={max(all_x):.1f}, image_width={w}")
        print(f"  Y: min={min(all_y):.1f}, max={max(all_y):.1f}, image_height={h}")

        if max(all_x) > w * 1.1 or max(all_y) > h * 1.1:
            print("\n⚠️  WARNING: Coordinates exceed image dimensions!")
        if min(all_x) < -w * 0.1 or min(all_y) < -h * 0.1:
            print("⚠️  WARNING: Coordinates are negative!")

if __name__ == '__main__':
    test_coordinates()

