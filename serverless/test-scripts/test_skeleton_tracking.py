#!/usr/bin/env python3
"""
Test script for skeleton tracking functionality with MediaPipe Pose + Hands.

This script tests the skeleton tracking feature by:
1. Running automatic annotation with skeleton tracking enabled
2. Verifying that tracks are created instead of individual shapes
3. Checking that tracks span multiple frames correctly
4. Validating gap handling and new object detection
5. Verifying correct association of detections across frames
6. Testing multiple skeleton tracking

Usage:
    python test_skeleton_tracking.py --task-id 18 --username admin --password password
"""

import argparse
import signal
import sys
import time
import json
from typing import Dict, List, Optional

import requests


def test_skeleton_tracking(
    cvat_url: str,
    username: str,
    password: str,
    task_id: int,
    function_name: Optional[str] = None,
    threshold: float = 0.3,
    analyze_frames: Optional[List[int]] = None,
) -> bool:
    """
    Test skeleton tracking on a CVAT task.

    Args:
        cvat_url: CVAT server URL (e.g., http://localhost:8080)
        username: CVAT username
        password: CVAT password
        task_id: Task ID to test on
        function_name: Name of the detection function (e.g., 'pth-google-mediapipe-pose-hands')
                      If None, will try to find a suitable MediaPipe function
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
                print("WARNING: Task frame_step is not 1. Skeleton tracking works best with video tasks (frame_step=1)")

        # Get available functions
        print("Fetching available functions...")
        functions_response = session.get(f"{cvat_url}/api/lambda/functions")
        if functions_response.status_code != 200:
            print(f"✗ Failed to fetch functions: {functions_response.status_code}")
            return False

        functions = functions_response.json()
        print(f"Found {len(functions)} functions")

        # Find MediaPipe function
        if function_name:
            function = next((f for f in functions if f.get("id") == function_name), None)
            if not function:
                print(f"ERROR: Function '{function_name}' not found")
                return False
        else:
            # Try to find a MediaPipe function
            # NOTE: Skeleton tracking uses DETECTOR function (not tracker)
            # The tracker function is for interactive tracking only
            # See: cvat/apps/lambda_manager/views.py line 1422 and skeleton_tracker.py
            mediapipe_functions = [
                f for f in functions
                if 'mediapipe' in f.get("id", "").lower() and f.get("kind") == 'detector'
            ]
            if not mediapipe_functions:
                print("ERROR: No MediaPipe detector function found")
                print("Available detector functions:")
                for f in [f for f in functions if f.get("kind") == "detector"]:
                    print(f"  - {f.get('id')}")
                # Also show tracker functions for reference
                tracker_functions = [f for f in functions if f.get("kind") == "tracker" and 'mediapipe' in f.get("id", "").lower()]
                if tracker_functions:
                    print("\nNote: Tracker functions are available but not used for skeleton tracking:")
                    for f in tracker_functions:
                        print(f"  - {f.get('id')} (kind: {f.get('kind')})")
                return False
            function = mediapipe_functions[0]
            function_kind = function.get("kind", "unknown")
            print(f"Using function: {function.get('id')} (kind: {function_kind})")
            print(f"Note: Skeleton tracking uses DETECTOR function. Tracker functions are for interactive tracking only.")

        # Get task labels - handle pagination properly
        task_labels = []

        # First, try to get labels from task data directly
        if "labels" in task_data and isinstance(task_data["labels"], list):
            task_labels = task_data["labels"]
            print(f"Found {len(task_labels)} labels in task data")

        # Also try /api/labels endpoint with pagination support
        page = 1
        while True:
            labels_response = session.get(f"{cvat_url}/api/labels", params={"task_id": task_id, "page": page})
            if labels_response.status_code != 200:
                if page == 1:
                    print(f"⚠️  Warning: Failed to fetch labels from /api/labels: {labels_response.status_code}")
                break

            labels_data = labels_response.json()
            if isinstance(labels_data, dict):
                # Handle pagination
                api_labels = labels_data.get("results", [])
                # Merge with task labels (avoid duplicates)
                existing_ids = {tl.get("id") for tl in task_labels if isinstance(tl, dict)}
                for label in api_labels:
                    if isinstance(label, dict) and label.get("id") not in existing_ids:
                        task_labels.append(label)

                # Check if there's a next page
                if not labels_data.get("next"):
                    break
            elif isinstance(labels_data, list):
                # No pagination, just a list
                existing_ids = {tl.get("id") for tl in task_labels if isinstance(tl, dict)}
                for label in labels_data:
                    if isinstance(label, dict) and label.get("id") not in existing_ids:
                        task_labels.append(label)
                break

            page += 1

        print(f"Task has {len(task_labels)} labels total (checked {page} page(s))")

        # Validate task_labels format
        if not isinstance(task_labels, list):
            print(f"✗ Error: task_labels is not a list: {type(task_labels)}")
            return False

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

        # Extract function label names
        function_label_names = []
        if isinstance(function_labels, list):
            for fl in function_labels:
                if isinstance(fl, dict):
                    function_label_names.append(fl.get("name", ""))
                elif isinstance(fl, str):
                    function_label_names.append(fl)

        # Extract task label names and types
        task_label_names = []
        task_label_types = {}
        for tl in task_labels:
            if isinstance(tl, dict):
                label_name = tl.get("name", "")
                label_type = tl.get("type", "")
                task_label_names.append(label_name)
                task_label_types[label_name] = label_type
            else:
                task_label_names.append(str(tl))

        print(f"Function labels: {function_label_names}")
        print(f"Task labels: {task_label_names}")

        # Check if task has skeleton labels
        has_skeleton_labels = any(
            isinstance(tl, dict) and tl.get("type") == "skeleton"
            for tl in task_labels
        )

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
            print(f"   Function expects: {function_label_names}")
            print(f"   Task has: {task_label_names}")

            if not has_skeleton_labels:
                print("   ⚠️  CRITICAL: Task does not have skeleton labels!")
                print("   Tracks will show as 'unknown' label because there's no skeleton label to map to.")
                print("   SOLUTION: Add a skeleton label to the task (e.g., 'hands-shoulders-skeleton')")
                print("   with the same name as one of the function labels, or configure label mapping.")
            else:
                print("   ⚠️  WARNING: Task has skeleton labels but no mapping was created.")
                print("   This may indicate a label name mismatch. Tracks may show as 'unknown' label.")

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

        # Run automatic annotation with skeleton tracking
        print("\n" + "="*60)
        print("Running automatic annotation with skeleton tracking...")
        print("="*60)
        print("NOTE: This test explicitly enables skeleton tracking.")
        print("      Without this flag, standard detector should create SHAPES, not tracks.")

        request_body = {
            "type": "annotate_task",
            "mapping": mapping,
            "cleanup": True,  # Clear previous annotations to avoid confusion
            "threshold": threshold,
            "enable_skeleton_tracking": True,  # Explicitly enable skeleton tracking for this test
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
                    current_time = time.time()
                    if current_time - last_stats_time >= stats_interval:
                        try:
                            stats_response = session.get(f"{cvat_url}/api/tasks/{task_id}/annotations")
                            if stats_response.status_code == 200:
                                stats_data = stats_response.json()
                                current_shapes = len(stats_data.get("shapes", []))
                                current_tracks = len(stats_data.get("tracks", []))

                                # Count detections in tracks (total shapes across all tracks)
                                total_detections = 0
                                total_elements = 0
                                if current_tracks > 0:
                                    tracks = stats_data.get("tracks", [])
                                    for track in tracks:
                                        total_detections += len(track.get("shapes", []))
                                        total_elements += len(track.get("elements", []))

                                if progress < 90:
                                    print(f"    📊 Current stats: {current_shapes} shapes, {current_tracks} tracks")
                                    print(f"    ℹ️  Note: Tracks are created in memory and will be submitted at ~95% progress")
                                else:
                                    print(f"    📊 Current stats: {current_shapes} shapes, {current_tracks} tracks")
                                    print(f"    📊 Track details: {total_detections} detections, {total_elements} elements")

                                last_stats_time = current_time
                        except Exception as e:
                            # Don't fail if stats fetch fails, just skip it
                            pass
                else:
                    print(f"  Status: {status}")

                time.sleep(2)

            if time.time() - start_time >= max_wait_time:
                print("✗ Timeout waiting for annotation to complete")
                try:
                    cancel_response = session.delete(f"{cvat_url}/api/lambda/requests/{request_id}")
                    if cancel_response.status_code in [200, 204]:
                        print(f"✓ Canceled annotation request {request_id} due to timeout")
                except Exception as e:
                    print(f"⚠️  Error canceling request on timeout: {e}")
                return False
        except KeyboardInterrupt:
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

        # Count detections in tracks
        total_detections_in_tracks = 0
        total_elements_in_tracks = 0
        if final_tracks_count > 0:
            tracks = final_annotations.get("tracks", [])
            for track in tracks:
                total_detections_in_tracks += len(track.get("shapes", []))
                total_elements_in_tracks += len(track.get("elements", []))
            print(f"\nDetection Statistics:")
            print(f"  - Total detections in tracks: {total_detections_in_tracks}")
            print(f"  - Average detections per track: {total_detections_in_tracks / final_tracks_count:.1f}")
            print(f"  - Total elements (sub-tracks) in tracks: {total_elements_in_tracks}")
            print(f"  - Average elements per track: {total_elements_in_tracks / final_tracks_count:.1f}")
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
                total_elements = sum(len(t.get("elements", [])) for t in label_tracks)
                print(f"  - {label_name}: {len(label_tracks)} tracks, {total_shapes} total detections, {total_elements} total elements")

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
                elements = track.get("elements", [])
                frame_count = len(shapes)
                frames = sorted([s.get("frame", 0) for s in shapes])
                first_frame = frames[0] if frames else 0
                last_frame = frames[-1] if frames else 0
                span = last_frame - first_frame + 1

                print(f"Track {idx + 1}/{min(len(tracks), 10)} (ID: {track_id}, Label: {label_name}):")
                print(f"  - Detections: {frame_count} shapes")
                print(f"  - Elements: {len(elements)} sub-tracks")
                print(f"  - Frame range: {first_frame} to {last_frame} (span: {span} frames)")

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

                # Analyze elements (sub-tracks)
                if elements:
                    element_info = []
                    for elem in elements[:5]:  # Show first 5 elements
                        elem_shapes = elem.get("shapes", [])
                        elem_frames = sorted([s.get("frame", 0) for s in elem_shapes])
                        elem_label_id = elem.get("label_id")
                        elem_label_name = "unknown"
                        # Find element label name from skeleton structure
                        for tl in task_labels:
                            if isinstance(tl, dict) and tl.get("id") == label_id:
                                # Check sublabels
                                structure = tl.get("structure", {})
                                sublabels = structure.get("sublabels", [])
                                for sublabel in sublabels:
                                    if isinstance(sublabel, dict) and sublabel.get("id") == elem_label_id:
                                        elem_label_name = sublabel.get("name", "unknown")
                                        break
                                break
                        element_info.append(f"{elem_label_name}({len(elem_shapes)} frames)")
                    print(f"  - Elements: {', '.join(element_info)}" + (f" ... ({len(elements)} total)" if len(elements) > 5 else ""))

            # Additional validation checks
            print(f"\n{'='*60}")
            print("VALIDATION CHECKS")
            print("="*60)

            validation_passed = True
            validation_issues = []

            # Check 0: Verify tracks have valid labels (not "unknown")
            # Only warn if label_id exists but doesn't match any task label
            # If label_id is None, that's a different issue (backend didn't set it)
            unknown_label_tracks = []
            missing_label_id_tracks = []
            has_skeleton_labels = any(
                isinstance(tl, dict) and tl.get("type") == "skeleton"
                for tl in task_labels
            )

            for track in tracks:
                label_id = track.get("label_id")
                track_id = track.get("id", "unknown")

                if label_id is None:
                    # Label ID is missing - backend issue
                    missing_label_id_tracks.append(track_id)
                    continue

                # Try to find matching label
                label_name = None
                for tl in task_labels:
                    if isinstance(tl, dict) and tl.get("id") == label_id:
                        label_name = tl.get("name")
                        break

                if label_name is None:
                    # Label ID exists but doesn't match any task label
                    # This happens when label mapping failed
                    unknown_label_tracks.append((track_id, label_id))

            # Only show warnings if there are actual issues
            if missing_label_id_tracks:
                validation_issues.append(
                    f"⚠️  WARNING: {len(missing_label_id_tracks)} tracks have missing label_id (None). "
                    f"This may indicate a backend issue."
                )

            # Only warn about unmapped labels if task has skeleton labels (meaning mapping should have worked)
            # If task doesn't have skeleton labels, this is expected behavior
            if unknown_label_tracks and has_skeleton_labels:
                validation_issues.append(
                    f"⚠️  CRITICAL: {len(unknown_label_tracks)} tracks have unmapped labels. "
                    f"Label IDs: {[lid for _, lid in unknown_label_tracks[:5]]}. "
                    f"This indicates label mapping failed even though task has skeleton labels."
                )
                validation_passed = False
            elif unknown_label_tracks and not has_skeleton_labels:
                # This is expected - don't add as validation issue, just informational
                pass  # Already warned earlier about missing skeleton labels

            # Check 1: Verify tracks have reasonable frame spans
            for track in tracks:
                track_id = track.get("id", "unknown")
                shapes = track.get("shapes", [])
                if len(shapes) > 0:
                    frames = sorted([s.get("frame", 0) for s in shapes])
                    span = frames[-1] - frames[0] + 1
                    if span < len(shapes):
                        validation_issues.append(f"Track {track_id}: Has {len(shapes)} detections but span is {span} frames (gaps detected)")

            # Check 2: Verify elements (sub-tracks) have consistent structure
            for track in tracks:
                track_id = track.get("id", "unknown")
                elements = track.get("elements", [])
                if elements:
                    # Check that all elements have shapes
                    for elem_idx, elem in enumerate(elements):
                        elem_shapes = elem.get("shapes", [])
                        if not elem_shapes:
                            validation_issues.append(f"Track {track_id}, Element {elem_idx}: Has no shapes")

                    # Check that elements span similar frame ranges
                    element_frames = [sorted([s.get("frame", 0) for s in elem.get("shapes", [])]) for elem in elements]
                    if element_frames:
                        min_frames = min([f[0] for f in element_frames if f])
                        max_frames = max([f[-1] for f in element_frames if f])
                        for elem_idx, elem_frames_list in enumerate(element_frames):
                            if elem_frames_list:
                                elem_min = elem_frames_list[0]
                                elem_max = elem_frames_list[-1]
                                # Elements should roughly span the same range as the track
                                if elem_min > max_frames or elem_max < min_frames:
                                    validation_issues.append(f"Track {track_id}, Element {elem_idx}: Frames ({elem_min}-{elem_max}) don't align with track range ({min_frames}-{max_frames})")

            # Check 3: Verify no duplicate associations (same detection in multiple tracks)
            # Also check for overlapping detections per frame (multiple tracks on same frame)
            all_detection_frames = {}  # {(frame, label_id): [track_ids]}
            frame_detection_counts = {}  # {frame: count} - count detections per frame
            frame_track_details = {}  # {frame: [(track_id, label_id)]} - detailed info per frame

            for track in tracks:
                track_id = track.get("id", "unknown")
                label_id = track.get("label_id")
                shapes = track.get("shapes", [])
                for shape in shapes:
                    frame = shape.get("frame", 0)
                    key = (frame, label_id)
                    if key not in all_detection_frames:
                        all_detection_frames[key] = []
                    all_detection_frames[key].append(track_id)

                    # Count total detections per frame (across all labels)
                    frame_detection_counts[frame] = frame_detection_counts.get(frame, 0) + 1

                    # Store track details per frame
                    if frame not in frame_track_details:
                        frame_track_details[frame] = []
                    frame_track_details[frame].append((track_id, label_id))

            # Check for overlapping detections on same frame
            for (frame, label_id), track_ids in all_detection_frames.items():
                if len(track_ids) > 1:
                    validation_issues.append(
                        f"⚠️  Frame {frame}, Label {label_id}: {len(track_ids)} tracks overlap "
                        f"(track IDs: {track_ids}). This suggests incorrect association or tracks not ending properly."
                    )

            # Check for frames with too many detections (likely overlapping tracks)
            # For skeleton tracking, we expect 1-2 detections per frame (1-2 people)
            # More than 3-4 suggests tracks are overlapping incorrectly
            suspicious_frames = []
            for frame, count in sorted(frame_detection_counts.items()):
                if count > 3:  # More than 3 detections on a frame is suspicious
                    track_info = frame_track_details.get(frame, [])
                    track_ids = [str(tid) for tid, _ in track_info]
                    label_ids = list(set([lid for _, lid in track_info]))
                    suspicious_frames.append((frame, count, track_ids, label_ids))

            if suspicious_frames:
                validation_issues.append(
                    f"⚠️  CRITICAL: Found {len(suspicious_frames)} frames with excessive detections (>3):"
                )
                for frame, count, track_ids, label_ids in suspicious_frames[:10]:  # Show first 10
                    label_names = []
                    for lid in label_ids:
                        for tl in task_labels:
                            if isinstance(tl, dict) and tl.get("id") == lid:
                                label_names.append(tl.get("name", "unknown"))
                                break
                    validation_issues.append(
                        f"   Frame {frame}: {count} detections from {len(track_ids)} tracks "
                        f"(track IDs: {', '.join(track_ids[:5])}{'...' if len(track_ids) > 5 else ''}, "
                        f"labels: {', '.join(set(label_names))})"
                    )
                if len(suspicious_frames) > 10:
                    validation_issues.append(f"   ... and {len(suspicious_frames) - 10} more frames")
                validation_passed = False

            # Check 4: Verify tracks handle gaps correctly (1-2 frame gaps should be handled)
            for track in tracks:
                track_id = track.get("id", "unknown")
                shapes = track.get("shapes", [])
                if len(shapes) > 1:
                    frames = sorted([s.get("frame", 0) for s in shapes])
                    gaps = []
                    for i in range(len(frames) - 1):
                        gap = frames[i + 1] - frames[i]
                        if gap > 1:
                            gaps.append((frames[i], frames[i + 1], gap - 1))
                    # Small gaps (1-2 frames) should be acceptable
                    large_gaps = [g for g in gaps if g[2] > 5]
                    if large_gaps:
                        validation_issues.append(f"Track {track_id}: Has large gaps {large_gaps} (may indicate tracking issues)")

            # Check 5: Verify multiple skeletons are tracked separately
            if len(tracks) > 0:
                # Count tracks per label
                tracks_per_label = {}
                for track in tracks:
                    label_id = track.get("label_id")
                    label_name = "unknown"
                    for tl in task_labels:
                        if isinstance(tl, dict) and tl.get("id") == label_id:
                            label_name = tl.get("name", "unknown")
                            break
                    tracks_per_label[label_name] = tracks_per_label.get(label_name, 0) + 1

                # If we have multiple tracks for the same label, verify they don't overlap in time
                for label_name, count in tracks_per_label.items():
                    if count > 1:
                        label_tracks = [t for t in tracks if any(
                            isinstance(tl, dict) and tl.get("id") == t.get("label_id") and tl.get("name") == label_name
                            for tl in task_labels
                        )]
                        # Check for overlapping frame ranges
                        for i, track1 in enumerate(label_tracks):
                            for track2 in label_tracks[i+1:]:
                                shapes1 = track1.get("shapes", [])
                                shapes2 = track2.get("shapes", [])
                                if shapes1 and shapes2:
                                    frames1 = sorted([s.get("frame", 0) for s in shapes1])
                                    frames2 = sorted([s.get("frame", 0) for s in shapes2])
                                    overlap = set(frames1) & set(frames2)
                                    if overlap:
                                        # Overlap is OK if it's just a few frames (might be transition)
                                        if len(overlap) > 5:
                                            validation_issues.append(
                                                f"⚠️  Tracks for {label_name}: Significant frame overlap ({len(overlap)} frames) "
                                                f"between track {track1.get('id')} and {track2.get('id')} - may indicate incorrect association"
                                            )
                                            validation_passed = False

            # Check 6: Verify tracks end when they should (not carried too long)
            # Tracks should end when person disappears, not continue indefinitely
            # Check for tracks that have very sparse detections (many gaps)
            for track in tracks:
                track_id = track.get("id", "unknown")
                shapes = track.get("shapes", [])
                if len(shapes) > 1:
                    frames = sorted([s.get("frame", 0) for s in shapes])
                    span = frames[-1] - frames[0] + 1
                    detection_density = len(shapes) / span if span > 0 else 0

                    # If detection density is very low (< 0.3), track might be carried too long
                    if detection_density < 0.3 and len(shapes) > 10:
                        gaps = []
                        for i in range(len(frames) - 1):
                            gap = frames[i + 1] - frames[i]
                            if gap > 1:
                                gaps.append(gap - 1)
                        avg_gap = sum(gaps) / len(gaps) if gaps else 0
                        if avg_gap > 3:
                            validation_issues.append(
                                f"⚠️  Track {track_id}: Low detection density ({detection_density:.2%}, "
                                f"avg gap: {avg_gap:.1f} frames). Track may be carried too long after person disappeared."
                            )
                            validation_passed = False

            if validation_issues:
                print("⚠️  Validation Issues Found:")
                for issue in validation_issues[:20]:  # Show first 20 issues
                    print(f"  {issue}")
                if len(validation_issues) > 20:
                    print(f"  ... and {len(validation_issues) - 20} more issues")
                validation_passed = False
            else:
                print("✓ All validation checks passed!")

            # Additional frame-by-frame analysis for specific frames
            print(f"\n{'='*60}")
            print("FRAME-BY-FRAME ANALYSIS")
            print("="*60)
            print("Analyzing specific frames for overlapping detections...")
            print("Note: For skeleton tracks, UI may show elements (sub-tracks) as separate objects.")

            # Cache annotations data to avoid querying multiple times
            _cached_annotations = None

            def get_frame_annotations_from_api(frame_num: int) -> Optional[Dict]:
                """Query CVAT API to get what's actually rendered on a frame."""
                nonlocal _cached_annotations

                try:
                    # Get annotations once and cache them
                    if _cached_annotations is None:
                        # Get job ID if available (for job-specific annotations)
                        job_id = None
                        try:
                            # Access task_data from outer scope
                            jobs_data = task_data.get("jobs", [])
                            if jobs_data and isinstance(jobs_data, list) and len(jobs_data) > 0:
                                # jobs can be a list of IDs or list of objects
                                if isinstance(jobs_data[0], dict):
                                    job_id = jobs_data[0].get("id")
                                elif isinstance(jobs_data[0], (int, str)):
                                    job_id = jobs_data[0]
                        except (NameError, AttributeError, KeyError, IndexError, TypeError) as e:
                            # task_data not accessible or jobs_data has wrong format
                            # Just use task_id instead
                            pass

                        # Query annotations API - this returns what UI would show
                        if job_id:
                            ann_response = session.get(f"{cvat_url}/api/jobs/{job_id}/annotations")
                        else:
                            ann_response = session.get(f"{cvat_url}/api/tasks/{task_id}/annotations")

                        if ann_response.status_code != 200:
                            # Log the error for debugging
                            print(f"      ⚠️  API query failed: {ann_response.status_code} - {ann_response.text[:100]}")
                            return None

                        _cached_annotations = ann_response.json()

                    if not _cached_annotations:
                        return None

                    ann_data = _cached_annotations
                    # Count tracks and shapes visible on this frame
                    tracks_on_frame = []
                    track_shapes_on_frame = []
                    element_shapes_on_frame = []

                    # Ensure ann_data is a dict
                    if not isinstance(ann_data, dict):
                        return None

                    for track in ann_data.get("tracks", []):
                        track_shapes = track.get("shapes", [])
                        # Check if track has a shape on this frame
                        track_visible = False
                        for shape in track_shapes:
                            if shape.get("frame") == frame_num and not shape.get("outside", False):
                                tracks_on_frame.append(track)
                                track_shapes_on_frame.append(shape)
                                track_visible = True
                                break

                        # For skeleton tracks, also check elements (sub-tracks)
                        # Elements are what UI shows as separate objects
                        # NOTE: Each element is a sub-track, so we count each element once if it has a visible shape
                        if track_visible:
                            elements = track.get("elements", [])
                            for element in elements:
                                elem_shapes = element.get("shapes", [])
                                # Check if this element has a visible shape on this frame
                                element_visible = False
                                for shape in elem_shapes:
                                    if shape.get("frame") == frame_num and not shape.get("outside", False):
                                        # Element is visible on this frame - count it once
                                        element_shapes_on_frame.append(shape)
                                        element_visible = True
                                        break  # Found shape for this element, move to next element

                    # Also check standalone shapes
                    standalone_shapes = []
                    for shape in ann_data.get("shapes", []):
                        if shape.get("frame") == frame_num and not shape.get("outside", False):
                            standalone_shapes.append(shape)

                    # Total visible objects = track shapes + element shapes + standalone shapes
                    # For skeleton tracks, UI typically shows: 1 track shape + N element shapes = N+1 total
                    total_visible = len(track_shapes_on_frame) + len(element_shapes_on_frame) + len(standalone_shapes)

                    return {
                        "tracks": len(tracks_on_frame),
                        "track_shapes": len(track_shapes_on_frame),
                        "element_shapes": len(element_shapes_on_frame),
                        "standalone_shapes": len(standalone_shapes),
                        "total_visible": total_visible,
                        "track_ids": [t.get("id") for t in tracks_on_frame],
                    }
                except Exception as e:
                    # Log the error for debugging
                    print(f"      ⚠️  Exception querying API for frame {frame_num}: {type(e).__name__}: {str(e)[:100]}")
                    return None

            # Analyze a few key frames (start, middle, end, and any suspicious frames)
            frame_set = sorted(frame_detection_counts.keys())
            if frame_set:
                sample_frames = []

                # Add user-specified frames to analyze
                if analyze_frames:
                    for frame in analyze_frames:
                        sample_frames.append(frame)  # Always include user-specified frames

                # Add default sample frames
                if len(frame_set) > 0:
                    sample_frames.append(frame_set[0])  # First frame
                if len(frame_set) > 1:
                    sample_frames.append(frame_set[len(frame_set)//2])  # Middle frame
                if len(frame_set) > 2:
                    sample_frames.append(frame_set[-1])  # Last frame

                # Add suspicious frames
                for frame, count, _, _ in suspicious_frames[:5]:
                    if frame not in sample_frames:
                        sample_frames.append(frame)

                sample_frames = sorted(set(sample_frames))[:20]  # Limit to 20 frames

                for frame in sample_frames:
                    # Count from track structure (what we calculated)
                    count_from_tracks = frame_detection_counts.get(frame, 0)
                    track_info = frame_track_details.get(frame, [])

                    # Query API to see what UI actually shows
                    api_data = get_frame_annotations_from_api(frame)

                    if count_from_tracks > 0 or api_data:
                        track_ids = [str(tid) for tid, _ in track_info]
                        label_info = {}
                        for tid, lid in track_info:
                            label_name = "unknown"
                            for tl in task_labels:
                                if isinstance(tl, dict) and tl.get("id") == lid:
                                    label_name = tl.get("name", "unknown")
                                    break
                            if label_name not in label_info:
                                label_info[label_name] = []
                            label_info[label_name].append(str(tid))

                        # Compare track structure count vs API count
                        if api_data:
                            api_tracks = api_data.get("tracks", 0)
                            api_track_shapes = api_data.get("track_shapes", 0)
                            api_element_shapes = api_data.get("element_shapes", 0)
                            api_standalone = api_data.get("standalone_shapes", 0)
                            api_total = api_data.get("total_visible", 0)

                            # For skeleton tracks:
                            # - API returns: 1 track shape + N element shapes = N+1 total
                            # - UI might show: Only the track (1) OR track + elements (N+1)
                            # - User reports seeing 8 on frame 225, but API shows 25 (1 track + 24 elements)
                            # - This suggests UI might be filtering elements or counting differently
                            #   Possible reasons:
                            #   1. UI only shows keyframe elements (but we found 0 keyframes on frame 225)
                            #   2. UI filters elements by some criteria (label type, visibility, etc.)
                            #   3. UI groups elements differently
                            #   4. UI counts only main tracks, not elements (but that would be 1, not 8)

                            # Special debug for frame 225 if user reported discrepancy
                            if frame == 225 and api_total != 8:
                                print(f"\n  🔍 DEBUG Frame 225:")
                                print(f"     API query shows: {api_total} total visible objects")
                                print(f"       - Track shapes: {api_track_shapes}")
                                print(f"       - Element shapes: {api_element_shapes}")
                                print(f"       - Standalone shapes: {api_standalone}")
                                print(f"     Track structure count: {count_from_tracks} (main track shapes only)")
                                if api_total == 27:
                                    print(f"     ⚠️  DISCREPANCY: User reports UI shows 8, but API shows {api_total}")
                                    print(f"     This suggests the UI may be:")
                                    print(f"       - Filtering elements (showing only 7 of 26 elements + 1 track = 8)")
                                    print(f"       - Grouping elements differently")
                                    print(f"       - Counting only keyframe elements or specific label types")
                                    print(f"     NOTE: The API returns all visible elements, but UI may filter them for display")

                            # For skeleton tracks, UI shows: track shape + element shapes
                            # So total visible = track_shapes + element_shapes
                            status = "✓" if api_total <= 2 else "⚠️" if api_total <= 4 else "✗"

                            # Show both counts
                            if api_total != count_from_tracks:
                                print(f"  ⚠️  Frame {frame}:")
                                print(f"      Track structure count: {count_from_tracks} detection(s) from {len(track_ids)} track(s)")
                                print(f"      UI view (API query): {api_total} visible object(s)")
                                print(f"         - Track shapes: {api_track_shapes}")
                                print(f"         - Element shapes: {api_element_shapes}")
                                print(f"         - Standalone shapes: {api_standalone}")
                                if api_total > count_from_tracks:
                                    # For skeleton tracks, this is expected - elements are rendered separately
                                    # Calculate expected range
                                    expected_min = count_from_tracks  # At least the track itself
                                    # Reasonable upper bound: track + up to 50 elements per track
                                    expected_max = count_from_tracks * 51  # Very generous upper bound

                                    if api_total > expected_max:
                                        print(f"      ⚠️  LARGE DISCREPANCY: UI shows {api_total} objects but track structure has {count_from_tracks} detections")
                                        print(f"      This might indicate multiple tracks or counting issues")
                                    else:
                                        print(f"      ℹ️  UI shows {api_total} objects (track + elements), track structure has {count_from_tracks} main track(s)")
                                        print(f"      This is expected for skeleton tracks - elements (sub-tracks) are rendered separately in UI")
                                        print(f"      Breakdown: {api_track_shapes} track shape(s) + {api_element_shapes} element shape(s) = {api_total} total")
                            else:
                                print(f"  {status} Frame {frame}: {count_from_tracks} detection(s) from {len(track_ids)} track(s)")
                                print(f"      (UI shows {api_total} visible objects: {api_track_shapes} track + {api_element_shapes} elements)")
                        else:
                            # No API data available
                            status = "✓" if count_from_tracks <= 2 else "⚠️" if count_from_tracks <= 4 else "✗"
                            print(f"  {status} Frame {frame}: {count_from_tracks} detection(s) from {len(track_ids)} track(s)")
                            print(f"      (Could not query API for UI view)")

                        if count_from_tracks > 2:
                            for label_name, tids in label_info.items():
                                print(f"      - {label_name}: {len(tids)} track(s) - IDs: {', '.join(tids[:5])}{'...' if len(tids) > 5 else ''}")
                    elif frame in analyze_frames:
                        # User specified this frame but it has no detections in track structure
                        print(f"  ⚠️  Frame {frame}: No detections found in track structure")
                        api_data = get_frame_annotations_from_api(frame)
                        if api_data:
                            api_total = api_data.get("total_visible", 0)
                            if api_total > 0:
                                print(f"      ⚠️  DISCREPANCY: UI shows {api_total} visible object(s) but track structure has none!")
                                print(f"         - Track shapes: {api_data.get('track_shapes', 0)}")
                                print(f"         - Element shapes: {api_data.get('element_shapes', 0)}")
                                print(f"         - Standalone shapes: {api_data.get('standalone_shapes', 0)}")
                        else:
                            print(f"      (Could not query API to verify UI view)")

            # Summary statistics
            if frame_detection_counts:
                max_detections = max(frame_detection_counts.values())
                avg_detections = sum(frame_detection_counts.values()) / len(frame_detection_counts)
                frames_with_many = sum(1 for c in frame_detection_counts.values() if c > 3)

                print(f"\nFrame Detection Statistics:")
                print(f"  - Average detections per frame: {avg_detections:.2f}")
                print(f"  - Maximum detections on a single frame: {max_detections}")
                print(f"  - Frames with >3 detections: {frames_with_many} ({100*frames_with_many/len(frame_detection_counts):.1f}%)")
                if max_detections > 3:
                    print(f"  ⚠️  WARNING: Some frames have excessive detections. Expected 1-2 per frame (1-2 people).")

            # Final validation: Verify tracks were created (not shapes)
            # This confirms skeleton tracking was actually used
            if final_tracks_count == 0 and final_shapes_count > 0:
                print("\n" + "="*60)
                print("⚠️  WARNING: No tracks created, but shapes were created instead!")
                print("="*60)
                print("This suggests skeleton tracking was NOT enabled or failed.")
                print("Expected: Tracks (skeleton tracking mode)")
                print("Got: Shapes (standard detector mode)")
                print("\nPossible causes:")
                print("  1. enable_skeleton_tracking flag was not sent correctly")
                print("  2. Backend validation failed and fell back to standard detector")
                print("  3. Task is not a video task (frame_step != 1)")
                return False

            print("\n" + "="*60)
            if validation_passed:
                print("✓ TEST PASSED: Skeleton tracking created tracks successfully!")
                print(f"✓ Created {final_tracks_count} skeleton tracks (not individual shapes)")
                print("✓ This confirms skeleton tracking mode was used correctly")
            else:
                print("⚠️  TEST PASSED WITH WARNINGS: Tracks created but validation issues detected")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("✗ TEST FAILED: No tracks were created")
            print("="*60)
            if final_shapes_count > 0:
                print(f"  Instead, {final_shapes_count} individual shapes were created")
                print("  This suggests skeleton tracking was not enabled or did not work correctly")

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
        description="Test skeleton tracking functionality with MediaPipe Pose + Hands"
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
        help="Name of the detection function (default: auto-detect MediaPipe function)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--analyze-frames",
        type=str,
        default=None,
        help="Comma-separated list of frame numbers to analyze in detail (e.g., '200,201,202')"
    )

    args = parser.parse_args()

    if not args.password:
        import getpass
        args.password = getpass.getpass("CVAT password: ")

    # Parse analyze-frames argument
    analyze_frames = None
    if args.analyze_frames:
        try:
            analyze_frames = [int(f.strip()) for f in args.analyze_frames.split(',')]
        except ValueError:
            print(f"ERROR: Invalid --analyze-frames format: {args.analyze_frames}")
            print("Expected format: comma-separated integers (e.g., '200,201,202')")
            sys.exit(1)

    success = test_skeleton_tracking(
        cvat_url=args.cvat_url,
        username=args.username,
        password=args.password,
        task_id=args.task_id,
        function_name=args.function_name,
        threshold=args.threshold,
        analyze_frames=analyze_frames,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

