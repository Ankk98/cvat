#!/usr/bin/env python3
"""
Test Frontend Mask Rendering
============================

This script simulates how CVAT's frontend renders masks to identify
format issues. It implements the frontend RLE decoding logic.

Note: Service outputs flattened pixels, CVAT backend converts to RLE,
then frontend renders the RLE. This script tests the final RLE format
that reaches the frontend (after backend processing).
"""

import numpy as np
import json
import requests
from typing import List, Tuple


def rle2mask_frontend(rle: List[float], width: int, height: int) -> np.ndarray:
    """
    Simulate CVAT frontend's rle2Mask function.

    Source: cvat-canvas/src/typescript/shared.ts:428-450
    """
    decoded = np.zeros((width * height, 4), dtype=np.uint8)  # RGBA
    length = len(rle)
    decoded_idx = 0
    value = 0
    i = 0

    # Process run lengths (everything except last 4 bbox values)
    while i < length - 4:
        count = int(rle[i])
        while count > 0:
            # Set alpha channel based on value (0 or 1)
            decoded[decoded_idx, 3] = value * 255  # Alpha channel
            decoded_idx += 1
            count -= 1
        i += 1
        value = abs(value - 1)  # Toggle between 0 and 1

    return decoded.reshape((height, width, 4))


def expand_channels_frontend(rle: List[float], r: int, g: int, b: int) -> np.ndarray:
    """
    Simulate CVAT frontend's expandChannels function.

    Source: cvat-canvas/src/typescript/shared.ts:427-454
    """
    # Extract bbox from last 4 values
    left, top, right, bottom = [int(x) for x in rle[-4:]]

    # Calculate dimensions
    width = right - left + 1  # Frontend uses +1 (inclusive)
    height = bottom - top + 1  # Frontend uses +1 (inclusive)

    print(f"  Frontend bbox: left={left}, top={top}, right={right}, bottom={bottom}")
    print(f"  Frontend dimensions: width={width}, height={height} (right-left+1, bottom-top+1)")

    # Decode RLE to mask
    mask = rle2mask_frontend(rle, width, height)

    return mask, (left, top, right, bottom)


def validate_mask_format(rle: List[float], image_width: int, image_height: int) -> Tuple[bool, str]:
    """
    Validate RLE format matches frontend expectations.
    """
    issues = []

    # Check minimum length
    if len(rle) < 6:
        issues.append(f"RLE has {len(rle)} values, need at least 6 (2 run lengths + 4 bbox)")
        return False, "; ".join(issues)

    # Check if values are integers (frontend might expect integers)
    if any(isinstance(x, float) and not x.is_integer() for x in rle):
        non_int = [x for x in rle if isinstance(x, float) and not x.is_integer()]
        issues.append(f"RLE contains non-integer floats: {non_int[:5]}")

    # Extract bbox
    left, top, right, bottom = [int(x) for x in rle[-4:]]

    # Frontend validation (object-utils.ts:42)
    width = right - left
    height = bottom - top

    if width < 0:
        issues.append(f"Invalid width: {width} (right={right} - left={left})")
    if height < 0:
        issues.append(f"Invalid height: {height} (bottom={bottom} - top={top})")
    if not isinstance(width, int) or not isinstance(height, int):
        issues.append(f"Width/height must be integers: width={width}, height={height}")

    # Frontend rendering dimensions (shared.ts:453)
    render_width = right - left + 1
    render_height = bottom - top + 1

    # Check if bbox is within image bounds
    if left < 0 or top < 0:
        issues.append(f"Bbox starts outside image: left={left}, top={top}")
    if right >= image_width:
        issues.append(f"Bbox extends beyond image width: right={right} >= {image_width}")
    if bottom >= image_height:
        issues.append(f"Bbox extends beyond image height: bottom={bottom} >= {image_height}")

    # Check run lengths sum matches mask size
    run_lengths = rle[:-4]
    total_pixels = sum(int(x) for x in run_lengths)
    expected_pixels = render_width * render_height

    if total_pixels != expected_pixels:
        issues.append(f"Run lengths sum ({total_pixels}) != expected pixels ({expected_pixels})")

    if issues:
        return False, "; ".join(issues)

    return True, "Valid"


def test_cvat_annotation(cvat_url: str, task_id: int, job_id: int, frame: int = 0):
    """Test a real CVAT annotation."""
    session = requests.Session()
    session.post(f"{cvat_url}/api/auth/login",
                 json={"username": "admin", "password": "password"})

    # Get task info for image dimensions
    task_response = session.get(f"{cvat_url}/api/tasks/{task_id}")
    task_data = task_response.json()

    # Get size - could be dict or int
    size = task_data.get("size", {})
    if isinstance(size, dict):
        image_width = size.get("width", 1920)
        image_height = size.get("height", 1080)
    elif isinstance(size, int):
        # If size is just a number, try to get from data
        data_info = task_data.get("data", {})
        if isinstance(data_info, dict):
            image_width = data_info.get("image_quality", 100)  # Fallback
            image_height = data_info.get("image_quality", 100)  # Fallback
        else:
            image_width = 1920
            image_height = 1080
    else:
        image_width = 1920
        image_height = 1080

    # Get annotations
    anno_response = session.get(f"{cvat_url}/api/jobs/{job_id}/annotations")
    anno_data = anno_response.json()

    shapes = [s for s in anno_data.get("shapes", []) if s.get("frame") == frame and s.get("type") == "mask"]

    print("=" * 80)
    print("FRONTEND MASK RENDERING TEST")
    print("=" * 80)
    print(f"\nImage dimensions: {image_width}x{image_height}")
    print(f"Found {len(shapes)} mask annotations for frame {frame}\n")

    for i, shape in enumerate(shapes[:3]):  # Test first 3
        print(f"\n{'=' * 80}")
        print(f"MASK #{i+1} (label_id={shape.get('label_id')})")
        print("=" * 80)

        points = shape.get("points", [])
        print(f"RLE length: {len(points)}")
        print(f"RLE value types: {set(type(x).__name__ for x in points[:10])}")
        print(f"First 10 values: {points[:10]}")
        print(f"Last 4 values (bbox): {points[-4:]}")

        # Validate format
        is_valid, message = validate_mask_format(points, image_width, image_height)
        print(f"\nValidation: {'✅ PASS' if is_valid else '❌ FAIL'}")
        if not is_valid:
            print(f"  Issues: {message}")
        else:
            print(f"  {message}")

        # Test frontend rendering
        try:
            mask, bbox = expand_channels_frontend(points, 255, 0, 0)  # Red color
            print(f"\n✅ Frontend rendering successful!")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask dtype: {mask.dtype}")
            print(f"  Non-zero pixels: {np.count_nonzero(mask[:, :, 3])}")
            print(f"  Mask coverage: {np.count_nonzero(mask[:, :, 3]) / mask.size * 4 * 100:.2f}%")
        except Exception as e:
            print(f"\n❌ Frontend rendering failed: {e}")
            import traceback
            traceback.print_exc()


def test_our_rle_format():
    """Test our RLE format generation."""
    print("\n" + "=" * 80)
    print("TESTING OUR RLE FORMAT")
    print("=" * 80)

    # Simulate a simple mask: 10x10 with a 5x5 square in the middle
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:7, 2:7] = True  # 5x5 square

    # Our RLE generation (simplified)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    tight_mask = mask[y_min:y_max+1, x_min:x_max+1]
    flat = tight_mask.ravel()

    # Generate RLE
    diff = np.diff(flat, prepend=[not flat[0]], append=[not flat[-1]])
    run_indices = np.nonzero(diff)[0]

    if flat[0]:
        run_lengths = np.diff(run_indices, prepend=[0])
    else:
        run_lengths = np.diff(run_indices)

    # Our format: [run_lengths..., x_min, y_min, x_max, y_max] (inclusive)
    rle = [int(x) for x in run_lengths.tolist()] + [int(x_min), int(y_min), int(x_max), int(y_max)]

    print(f"\nGenerated RLE: length={len(rle)}")
    print(f"Run lengths: {run_lengths.tolist()}")
    print(f"Bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
    print(f"Full RLE: {rle}")

    # Test frontend rendering
    try:
        mask_rendered, bbox = expand_channels_frontend(rle, 255, 0, 0)
        print(f"\n✅ Frontend rendering successful!")
        print(f"  Rendered mask shape: {mask_rendered.shape}")
        print(f"  Expected: ({y_max-y_min+1}, {x_max-x_min+1}, 4)")
    except Exception as e:
        print(f"\n❌ Frontend rendering failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test frontend mask rendering")
    parser.add_argument("--cvat-url", default="http://localhost:8080", help="CVAT URL")
    parser.add_argument("--task-id", type=int, help="Task ID to test")
    parser.add_argument("--job-id", type=int, help="Job ID to test")
    parser.add_argument("--frame", type=int, default=0, help="Frame to test")

    args = parser.parse_args()

    # Test our format first
    test_our_rle_format()

    # Test real CVAT annotations if provided
    if args.task_id and args.job_id:
        test_cvat_annotation(args.cvat_url, args.task_id, args.job_id, args.frame)
    else:
        print("\n" + "=" * 80)
        print("To test real CVAT annotations, run:")
        print("  python3 test_frontend_mask_rendering.py --task-id 18 --job-id 14")

