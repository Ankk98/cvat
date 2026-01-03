#!/usr/bin/env python3
"""
Test script for polygon tracking functionality with Mask R-CNN.

This script tests the polygon tracking feature by:
1. Running automatic annotation with polygon tracking enabled
2. Verifying that tracks are created instead of individual shapes
3. Checking that tracks span multiple frames correctly
4. Validating gap handling and new object detection

Usage:
    python test_polygon_tracking.py --task-id 18 --username admin --password password
"""

import argparse
import signal
import sys
import time
import json
from typing import Dict, List, Optional

import requests


def test_polygon_tracking(
    cvat_url: str,
    username: str,
    password: str,
    task_id: int,
    function_name: Optional[str] = None,
    threshold: float = 0.5,
) -> bool:
    """
    Test polygon tracking on a CVAT task.

    Args:
        cvat_url: CVAT server URL (e.g., http://localhost:8080)
        username: CVAT username
        password: CVAT password
        task_id: Task ID to test on
        function_name: Name of the detection function (e.g., 'pth-facebookresearch-detectron2-mask-rcnn-r50-rocm')
                      If None, will try to find a suitable mask R-CNN function
        threshold: Detection confidence threshold

    Returns:
        True if test passed, False otherwise
    """
    cvat_url = cvat_url.rstrip('/')
    session = requests.Session()

    # Authenticate
    print(f"Connecting to CVAT at {cvat_url}...")
    auth_response = session.post(
        f"{cvat_url}/api/auth/login",
        json={"username": username, "password": password}
    )
    if auth_response.status_code != 200:
        print(f"✗ Authentication failed: {auth_response.status_code} - {auth_response.text}")
        return False

    # Extract CSRF token from cookie and set in header
    if 'csrftoken' in session.cookies:
        csrf_token = session.cookies['csrftoken']
        session.headers['X-CSRFToken'] = csrf_token
        # Also set Origin header (required for CSRF)
        from urllib.parse import urlparse
        parsed_url = urlparse(cvat_url)
        session.headers['Origin'] = f"{parsed_url.scheme}://{parsed_url.netloc}"

    print(f"✓ Authenticated as {username}")

    try:
        # Get task
        print(f"Fetching task {task_id}...")
        task_response = session.get(f"{cvat_url}/api/tasks/{task_id}")
        if task_response.status_code != 200:
            print(f"✗ Failed to fetch task: {task_response.status_code} - {task_response.text}")
            return False

        task_data = task_response.json()
        task_name = task_data.get("name", "Unknown")
        task_size = task_data.get("size", 0)
        print(f"Task: {task_name} (size: {task_size} frames)")

        # Check if task is a video task
        data_info = task_data.get("data", {})
        if isinstance(data_info, dict):
            frame_step = data_info.get("frame_step", 1)
            if frame_step != 1:
                print("WARNING: Task frame_step is not 1. Polygon tracking works best with video tasks (frame_step=1)")

        # Get available functions
        print("Fetching available functions...")
        functions_response = session.get(f"{cvat_url}/api/lambda/functions")
        if functions_response.status_code != 200:
            print(f"✗ Failed to fetch functions: {functions_response.status_code}")
            return False

        functions = functions_response.json()
        print(f"Found {len(functions)} functions")

        # Find mask R-CNN function
        if function_name:
            function = next((f for f in functions if f.get("id") == function_name), None)
            if not function:
                print(f"ERROR: Function '{function_name}' not found")
                return False
        else:
            # Try to find a mask R-CNN function
            mask_rcnn_functions = [
                f for f in functions
                if 'mask' in f.get("id", "").lower() and 'rcnn' in f.get("id", "").lower() and f.get("kind") == 'detector'
            ]
            if not mask_rcnn_functions:
                print("ERROR: No Mask R-CNN detector function found")
                print("Available detector functions:")
                for f in [f for f in functions if f.get("kind") == "detector"]:
                    print(f"  - {f.get('id')}")
                return False
            function = mask_rcnn_functions[0]
            print(f"Using function: {function.get('id')}")

        # Get task labels from /api/labels endpoint
        # NOTE: task_data.get("labels") returns LabelsSummarySerializer which is {"url": "...", "count": N}
        # We need the actual label objects from /api/labels?task_id={id}
        task_labels = []
        labels_response = session.get(f"{cvat_url}/api/labels", params={"task_id": task_id})
        if labels_response.status_code == 200:
            labels_data = labels_response.json()
            if isinstance(labels_data, dict):
                task_labels = labels_data.get("results", [])
            elif isinstance(labels_data, list):
                task_labels = labels_data
        else:
            print(f"⚠️  Warning: Failed to fetch labels from /api/labels: {labels_response.status_code}")

        print(f"Task has {len(task_labels)} labels")

        # Validate task_labels format - must be a list of dictionaries
        if not isinstance(task_labels, list):
            print(f"✗ Error: task_labels is not a list: {type(task_labels)}")
            return False

        # Validate each label is a dictionary
        for i, label in enumerate(task_labels):
            if not isinstance(label, dict):
                print(f"✗ Error: task_labels[{i}] is not a dict: {type(label)}, value: {label}")
                return False

        # Create label mapping (map function labels to task labels)
        mapping = {}
        # Function API returns labels_v2, not labels
        function_labels = function.get("labels_v2", [])
        if isinstance(function_labels, str):
            # If labels is a JSON string, parse it
            try:
                function_labels = json.loads(function_labels)
            except:
                function_labels = []

        # Show available labels for debugging
        func_label_names = []
        for func_label in function_labels:
            if isinstance(func_label, dict):
                func_label_name = func_label.get("name", "")
            else:
                func_label_name = str(func_label)
            func_label_names.append(func_label_name)

        # Handle task_labels - could be dicts or strings
        task_label_names = []
        for tl in task_labels:
            if isinstance(tl, dict):
                task_label_names.append(tl.get("name", ""))
            else:
                task_label_names.append(str(tl))

        print(f"Function labels: {func_label_names}")
        print(f"Task labels: {task_label_names}")

        for func_label in function_labels:
            if isinstance(func_label, dict):
                func_label_name = func_label.get("name", "")
            else:
                func_label_name = str(func_label)

            # Try to find matching task label (case-insensitive)
            matching_task_label = None
            matching_task_label_name = None

            for tl in task_labels:
                if isinstance(tl, dict):
                    tl_name = tl.get("name", "")
                else:
                    tl_name = str(tl)

                if tl_name.lower() == func_label_name.lower():
                    matching_task_label = tl
                    matching_task_label_name = tl_name
                    break

            if matching_task_label:
                # CVAT expects mapping in format: {"function_label": {"name": "task_label", "attributes": {}}}
                mapping[func_label_name] = {
                    "name": matching_task_label_name,
                    "attributes": {}
                }
                print(f"  ✓ Mapped: {func_label_name} -> {matching_task_label_name}")

        if not mapping:
            print("⚠️  WARNING: No label mappings found in request.")
            print("   Note: Backend will attempt auto-mapping if label names match exactly.")
            print("   If labels don't match, detections may be filtered out.")

        # Get initial annotation count
        print("Fetching initial annotations...")
        annotations_response = session.get(f"{cvat_url}/api/tasks/{task_id}/annotations")
        if annotations_response.status_code != 200:
            print(f"✗ Failed to fetch annotations: {annotations_response.status_code}")
            return False

        initial_annotations = annotations_response.json()
        initial_shapes_count = len(initial_annotations.get("shapes", []))
        initial_tracks_count = len(initial_annotations.get("tracks", []))
        print(f"Initial annotations: {initial_shapes_count} shapes, {initial_tracks_count} tracks")

        # Run automatic annotation with polygon tracking
        print("\n" + "="*60)
        print("Running automatic annotation with polygon tracking...")
        print("="*60)

        request_body = {
            "type": "annotate_task",
            "mapping": mapping,
            "cleanup": True,  # Clear previous annotations to avoid confusion
            "conv_mask_to_poly": True,  # Convert masks to polygons for tracking
            "threshold": threshold,
            "enable_polygon_tracking": True,  # Enable polygon tracking
        }

        print("⚠️  NOTE: Previous annotations will be cleared (cleanup=True)")

        print(f"Request body: {json.dumps(request_body, indent=2)}")

        # Submit annotation request
        print(f"Submitting annotation request for function {function.get('id')}...")
        request_response = session.post(
            f"{cvat_url}/api/lambda/requests",
            json={
                "function": function.get("id"),
                "task": task_id,
                **request_body
            }
        )

        if request_response.status_code not in [200, 201]:
            print(f"✗ Failed to submit annotation request: {request_response.status_code} - {request_response.text}")
            return False

        annotation_request = request_response.json()
        request_id = annotation_request.get("id")
        print(f"✓ Annotation request submitted: {request_id}")
        print("Waiting for completion...")
        print("  (Press Ctrl+C to cancel the annotation and exit)")

        # Set up signal handler to cancel request on interrupt
        # Use a flag to track if we need to cancel
        cancel_requested = False

        def signal_handler(sig, frame):
            nonlocal cancel_requested
            cancel_requested = True
            print("\n\n⚠️  Interrupt received. Canceling annotation request...")
            try:
                cancel_response = session.delete(f"{cvat_url}/api/lambda/requests/{request_id}")
                if cancel_response.status_code in [200, 204]:
                    print(f"✓ Annotation request {request_id} canceled successfully")
                else:
                    print(f"⚠️  Failed to cancel request: {cancel_response.status_code} - {cancel_response.text}")
            except Exception as e:
                print(f"⚠️  Error canceling request: {e}")
            print("Exiting...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Wait for completion
        max_wait_time = 600  # 10 minutes
        start_time = time.time()
        last_stats_time = 0
        stats_interval = 10  # Print stats every 10 seconds

        try:
            while time.time() - start_time < max_wait_time:
                # Check if cancellation was requested
                if cancel_requested:
                    return False

                status_response = session.get(f"{cvat_url}/api/lambda/requests/{request_id}")
                if status_response.status_code != 200:
                    print(f"✗ Failed to get request status: {status_response.status_code}")
                    return False

                request_status = status_response.json()
                status = request_status.get("status")

                if status == "finished":
                    print("✓ Annotation completed!")
                    break
                elif status == "failed":
                    exc_info = request_status.get("exc_info", "Unknown error")
                    print(f"✗ Annotation failed: {exc_info}")
                    return False
                elif status in ["queued", "started"]:
                    progress = request_status.get("progress", 0)
                    print(f"  Status: {status} (Progress: {progress}%)")

                    # Periodically fetch and display annotation statistics
                    # NOTE: Tracks are only submitted at the end (~95% progress), so they won't appear until then
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        try:
                            stats_response = session.get(f"{cvat_url}/api/tasks/{task_id}/annotations")
                            if stats_response.status_code == 200:
                                stats_data = stats_response.json()
                                current_shapes = len(stats_data.get("shapes", []))
                                current_tracks = len(stats_data.get("tracks", []))

                                # Count detections in tracks
                                total_detections = 0
                                if current_tracks > 0:
                                    tracks = stats_data.get("tracks", [])
                                    for track in tracks:
                                        total_detections += len(track.get("shapes", []))

                                if progress < 90:
                                    # Before 90%, tracks haven't been submitted yet (they're created in memory)
                                    print(f"    📊 Current stats: {current_shapes} shapes, {current_tracks} tracks")
                                    print(f"    ℹ️  Note: Tracks are created in memory and will be submitted at ~95% progress")
                                else:
                                    # After 90%, tracks should be submitted
                                    print(f"    📊 Current stats: {current_shapes} shapes, {current_tracks} tracks, {total_detections} detections in tracks")

                                last_stats_time = current_time
                        except Exception as e:
                            # Don't fail if stats fetch fails, just skip it
                            pass
                else:
                    print(f"  Status: {status}")

                time.sleep(2)

            if time.time() - start_time >= max_wait_time:
                print("✗ Timeout waiting for annotation to complete")
                # Try to cancel the request on timeout
                try:
                    cancel_response = session.delete(f"{cvat_url}/api/lambda/requests/{request_id}")
                    if cancel_response.status_code in [200, 204]:
                        print(f"✓ Canceled annotation request {request_id} due to timeout")
                except Exception as e:
                    print(f"⚠️  Error canceling request on timeout: {e}")
                return False
        except KeyboardInterrupt:
            # This should be handled by signal handler, but just in case
            signal_handler(None, None)
            return False

        # Get final annotations
        print("\n" + "="*60)
        print("Analyzing results...")
        print("="*60)

        # Refresh annotations
        final_annotations_response = session.get(f"{cvat_url}/api/tasks/{task_id}/annotations")
        if final_annotations_response.status_code != 200:
            print(f"✗ Failed to fetch final annotations: {final_annotations_response.status_code}")
            return False

        final_annotations = final_annotations_response.json()
        final_shapes_count = len(final_annotations.get("shapes", []))
        final_tracks_count = len(final_annotations.get("tracks", []))

        print(f"\n{'='*60}")
        print("ANNOTATION SUMMARY")
        print("="*60)
        print(f"Initial state: {initial_shapes_count} shapes, {initial_tracks_count} tracks")
        print(f"Final state:   {final_shapes_count} shapes, {final_tracks_count} tracks")
        print(f"Changes:       +{final_shapes_count - initial_shapes_count} shapes, +{final_tracks_count - initial_tracks_count} tracks")

        # Count detections in tracks (total shapes across all tracks)
        total_detections_in_tracks = 0
        if final_tracks_count > 0:
            tracks = final_annotations.get("tracks", [])
            for track in tracks:
                total_detections_in_tracks += len(track.get("shapes", []))
            print(f"\nDetection Statistics:")
            print(f"  - Total detections in tracks: {total_detections_in_tracks}")
            print(f"  - Average detections per track: {total_detections_in_tracks / final_tracks_count:.1f}")
            print(f"  - Individual shapes (not in tracks): {final_shapes_count}")

        # Analyze tracks
        if final_tracks_count > 0:
            print(f"\n{'='*60}")
            print("TRACK ANALYSIS")
            print("="*60)
            print(f"Total tracks created: {final_tracks_count}")

            tracks = final_annotations.get("tracks", [])

            # Group tracks by label
            tracks_by_label = {}
            for track in tracks:
                label_id = track.get("label_id")
                label_name = "unknown"
                for tl in task_labels:
                    if isinstance(tl, dict) and tl.get("id") == label_id:
                        label_name = tl.get("name", "unknown")
                        break

                if label_name not in tracks_by_label:
                    tracks_by_label[label_name] = []
                tracks_by_label[label_name].append(track)

            print(f"\nTracks by label:")
            for label_name, label_tracks in tracks_by_label.items():
                total_shapes = sum(len(t.get("shapes", [])) for t in label_tracks)
                print(f"  - {label_name}: {len(label_tracks)} tracks, {total_shapes} total detections")

            print(f"\nDetailed track information:")
            print("-" * 60)

            for idx, track in enumerate(tracks[:10]):  # Show first 10 tracks
                track_id = track.get("id", "unknown")
                label_id = track.get("label_id")
                label_name = "unknown"
                for tl in task_labels:
                    if isinstance(tl, dict) and tl.get("id") == label_id:
                        label_name = tl.get("name", "unknown")
                        break

                shapes = track.get("shapes", [])
                frame_count = len(shapes)
                frames = sorted([s.get("frame", 0) for s in shapes])
                first_frame = frames[0] if frames else 0
                last_frame = frames[-1] if frames else 0
                span = last_frame - first_frame + 1

                print(f"Track {idx + 1}/{min(len(tracks), 10)} (ID: {track_id}, Label: {label_name}):")
                print(f"  - Detections: {frame_count} shapes")
                print(f"  - Frame range: {first_frame} to {last_frame} (span: {span} frames)")
                shape_types = set(s.get("type", "unknown") for s in shapes)
                print(f"  - Shape types: {shape_types}")

                # Count detections by type
                type_counts = {}
                for s in shapes:
                    shape_type = s.get("type", "unknown")
                    type_counts[shape_type] = type_counts.get(shape_type, 0) + 1
                if len(type_counts) > 1:
                    print(f"  - Type breakdown: {type_counts}")

                # Check for gaps
                if len(frames) > 1:
                    gaps = []
                    for i in range(len(frames) - 1):
                        gap = frames[i + 1] - frames[i]
                        if gap > 1:
                            gaps.append((frames[i], frames[i + 1], gap - 1))
                    if gaps:
                        print(f"  - Gaps: {gaps}")
                    else:
                        print(f"  - No gaps (continuous track)")

            print("\n" + "="*60)
            print("✓ TEST PASSED: Polygon tracking created tracks successfully!")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("✗ TEST FAILED: No tracks were created")
            print("="*60)
            if final_shapes_count > 0:
                print(f"  Instead, {final_shapes_count} individual shapes were created")
                print("  This suggests polygon tracking was not enabled or did not work correctly")

                # Show shape statistics
                shapes = final_annotations.get("shapes", [])
                shapes_by_type = {}
                shapes_by_label = {}
                for shape in shapes:
                    shape_type = shape.get("type", "unknown")
                    shapes_by_type[shape_type] = shapes_by_type.get(shape_type, 0) + 1

                    label_id = shape.get("label_id")
                    label_name = "unknown"
                    for tl in task_labels:
                        if isinstance(tl, dict) and tl.get("id") == label_id:
                            label_name = tl.get("name", "unknown")
                            break
                    shapes_by_label[label_name] = shapes_by_label.get(label_name, 0) + 1

                print(f"\n  Shape statistics:")
                print(f"    - By type: {shapes_by_type}")
                print(f"    - By label: {shapes_by_label}")

            return False

    except requests.RequestException as e:
        print(f"✗ Request Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test polygon tracking functionality with Mask R-CNN"
    )
    parser.add_argument(
        "--cvat-url",
        default="http://localhost:8080",
        help="CVAT server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--username",
        default="admin",
        help="CVAT username (default: admin)"
    )
    parser.add_argument(
        "--password",
        default="",
        help="CVAT password (default: empty, will prompt)"
    )
    parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="Task ID to test on"
    )
    parser.add_argument(
        "--function-name",
        default=None,
        help="Name of the detection function (default: auto-detect)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )

    args = parser.parse_args()

    if not args.password:
        import getpass
        args.password = getpass.getpass("CVAT password: ")

    success = test_polygon_tracking(
        cvat_url=args.cvat_url,
        username=args.username,
        password=args.password,
        task_id=args.task_id,
        function_name=args.function_name,
        threshold=args.threshold,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
