#!/usr/bin/env python3
"""
Check CVAT Annotations Format
==============================

Script to inspect how annotations are stored in CVAT and verify
if they're masks or bounding boxes.
"""

import argparse
import requests
import json
from typing import Dict, List, Any
import sys


class CVATAnnotationChecker:
    def __init__(self, cvat_url: str, username: str = "admin", password: str = "password"):
        self.cvat_url = cvat_url.rstrip('/')
        self.username = username
        self.password = password
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self):
        """Authenticate with CVAT."""
        auth_url = f"{self.cvat_url}/api/auth/login"
        response = self.session.post(
            auth_url,
            json={"username": self.username, "password": self.password}
        )
        if response.status_code != 200:
            raise Exception(f"Authentication failed: {response.status_code} - {response.text}")
        print(f"✅ Authenticated as {self.username}")

    def get_annotations(self, task_id: int, job_id: int) -> Dict[str, Any]:
        """Fetch annotations for a task/job."""
        annotations_url = f"{self.cvat_url}/api/jobs/{job_id}/annotations"
        response = self.session.get(annotations_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch annotations: {response.status_code} - {response.text}")
        return response.json()

    def analyze_annotations(self, annotations: Dict[str, Any], frame: int = None, label_map: Dict = None):
        """Analyze annotation format."""
        print("\n" + "=" * 80)
        print("ANNOTATION ANALYSIS")
        print("=" * 80)

        shapes = annotations.get("shapes", [])
        if not shapes:
            print("⚠️  No shapes found in annotations")
            return

        # Filter by frame if specified
        if frame is not None:
            shapes = [s for s in shapes if s.get("frame") == frame]
            print(f"\n📊 Analyzing frame {frame} ({len(shapes)} shapes)")
        else:
            print(f"\n📊 Analyzing all frames ({len(shapes)} total shapes)")

        # Group by type
        by_type = {}
        for shape in shapes:
            shape_type = shape.get("type", "unknown")
            if shape_type not in by_type:
                by_type[shape_type] = []
            by_type[shape_type].append(shape)

        print(f"\n📈 Shape types found: {list(by_type.keys())}")
        print(f"   Counts: {[(t, len(s)) for t, s in by_type.items()]}")

        # Analyze each type
        for shape_type, type_shapes in by_type.items():
            print(f"\n{'=' * 80}")
            print(f"TYPE: {shape_type.upper()} ({len(type_shapes)} shapes)")
            print("=" * 80)

            for i, shape in enumerate(type_shapes[:5]):  # Show first 5
                print(f"\n  Shape #{i+1} (frame {shape.get('frame', 'N/A')}):")
                label_id = shape.get('label_id', 'N/A')
                label_name = shape.get('label_name', 'N/A')
                if label_map and label_id in label_map:
                    label_info = label_map[label_id]
                    label_name = label_info.get('name', label_name)
                    label_type = label_info.get('type', 'N/A')
                    print(f"    Label ID: {label_id} -> Name: {label_name}, Type: {label_type}")
                    if label_type != 'mask':
                        print(f"    ⚠️  CRITICAL: Label type is '{label_type}', not 'mask'! This will cause box rendering!")
                else:
                    print(f"    Label: {label_id} (name: {label_name})")
                print(f"    Source: {shape.get('source', 'N/A')}")
                print(f"    Points length: {len(shape.get('points', []))}")

                points = shape.get("points", [])
                if points:
                    if shape_type == "mask":
                        # Mask format: [run_lengths..., left, top, right, bottom]
                        if len(points) >= 4:
                            bbox = points[-4:]
                            run_lengths = points[:-4]
                            print(f"    Bbox (last 4): {bbox}")
                            print(f"    Run lengths count: {len(run_lengths)}")
                            print(f"    First 10 run lengths: {run_lengths[:10]}")

                            # Validate bbox
                            left, top, right, bottom = bbox
                            width = right - left
                            height = bottom - top
                            print(f"    Bbox dimensions: {width}x{height} (left={left}, top={top}, right={right}, bottom={bottom})")

                            # Check if valid
                            if width < 0 or height < 0:
                                print(f"    ⚠️  INVALID: Negative dimensions!")
                            if len(points) < 6:
                                print(f"    ⚠️  INVALID: Less than 6 values (need at least 2 run lengths + 4 bbox)")
                    elif shape_type == "rectangle":
                        # Rectangle format: [x1, y1, x2, y2]
                        if len(points) == 4:
                            x1, y1, x2, y2 = points
                            width = x2 - x1
                            height = y2 - y1
                            print(f"    Rectangle: ({x1}, {y1}) to ({x2}, {y2})")
                            print(f"    Dimensions: {width}x{height}")
                    else:
                        print(f"    Points: {points[:20]}..." if len(points) > 20 else f"    Points: {points}")

        # Check for mask-specific issues
        if "mask" in by_type:
            print(f"\n{'=' * 80}")
            print("MASK VALIDATION")
            print("=" * 80)

            mask_shapes = by_type["mask"]
            invalid_masks = []
            for shape in mask_shapes:
                points = shape.get("points", [])
                if len(points) < 6:
                    invalid_masks.append((shape.get("frame"), "Less than 6 values"))
                elif len(points) >= 4:
                    left, top, right, bottom = points[-4:]
                    width = right - left
                    height = bottom - top
                    if width < 0 or height < 0:
                        invalid_masks.append((shape.get("frame"), f"Invalid dimensions: {width}x{height}"))

            if invalid_masks:
                print(f"⚠️  Found {len(invalid_masks)} invalid masks:")
                for frame, reason in invalid_masks[:10]:
                    print(f"    Frame {frame}: {reason}")
            else:
                print("✅ All masks have valid format")

    def check_task_labels(self, task_id: int):
        """Check task label configuration."""
        task_url = f"{self.cvat_url}/api/tasks/{task_id}"
        response = self.session.get(task_url)
        if response.status_code != 200:
            print(f"⚠️  Could not fetch task info: {response.status_code}")
            return

        task_data = response.json()
        labels = task_data.get("labels", [])

        print(f"\n{'=' * 80}")
        print("TASK LABEL CONFIGURATION")
        print("=" * 80)
        print(f"Total labels: {len(labels)}")

        # Also try to get labels from the labels endpoint
        labels_url = f"{self.cvat_url}/api/tasks/{task_id}/labels"
        labels_response = self.session.get(labels_url)
        if labels_response.status_code == 200:
            labels_data = labels_response.json()
            if isinstance(labels_data, list):
                labels = labels_data
            elif isinstance(labels_data, dict):
                labels = labels_data.get("results", labels)

        label_map = {}
        for label in labels:
            if isinstance(label, dict):
                label_id = label.get('id')
                label_name = label.get('name', 'N/A')
                label_type = label.get('type', 'N/A')
                label_map[label_id] = {'name': label_name, 'type': label_type}
                print(f"\n  Label: {label_name}")
                print(f"    ID: {label_id}")
                print(f"    Type: {label_type}")
                if label_type != 'mask':
                    print(f"    ⚠️  WARNING: Label type is '{label_type}', not 'mask'!")
            else:
                print(f"\n  Label: {label} (raw format)")

        return label_map

    def check_function_metadata(self, task_id: int):
        """Check if function metadata matches task labels."""
        functions_url = f"{self.cvat_url}/api/lambda/functions"
        response = self.session.get(functions_url)
        if response.status_code != 200:
            print(f"⚠️  Could not fetch functions: {response.status_code}")
            return

        response_data = response.json()
        if isinstance(response_data, dict):
            functions = response_data.get("results", [])
        else:
            functions = response_data if isinstance(response_data, list) else []

        print(f"\n{'=' * 80}")
        print("LAMBDA FUNCTIONS")
        print("=" * 80)

        for func in functions:
            if isinstance(func, dict):
                func_name = func.get("name", "")
                if "mask" in func_name.lower() or "rcnn" in func_name.lower():
                    print(f"\n  Function: {func_name}")
                    print(f"    Kind: {func.get('kind', 'N/A')}")
                    spec = func.get("spec", {})
                    if isinstance(spec, dict):
                        labels = spec.get("labels", [])
                        print(f"    Labels in spec: {len(labels)}")
                        for label in labels[:5]:
                            if isinstance(label, dict):
                                print(f"      - {label.get('name', 'N/A')}: type={label.get('type', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="Check CVAT annotations format")
    parser.add_argument("--cvat-url", default="http://localhost:8080", help="CVAT URL")
    parser.add_argument("--task-id", type=int, required=True, help="Task ID")
    parser.add_argument("--job-id", type=int, required=True, help="Job ID")
    parser.add_argument("--frame", type=int, help="Specific frame to analyze")
    parser.add_argument("--username", default="admin", help="CVAT username")
    parser.add_argument("--password", default="password", help="CVAT password")

    args = parser.parse_args()

    try:
        checker = CVATAnnotationChecker(args.cvat_url, args.username, args.password)

        # Check task labels
        label_map = checker.check_task_labels(args.task_id)

        # Check function metadata
        checker.check_function_metadata(args.task_id)

        # Get and analyze annotations
        annotations = checker.get_annotations(args.task_id, args.job_id)
        checker.analyze_annotations(annotations, args.frame, label_map)

        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print("=" * 80)

        # Final diagnosis
        shapes = annotations.get("shapes", [])
        mask_shapes = [s for s in shapes if s.get("type") == "mask"]

        if mask_shapes:
            print(f"✅ Found {len(mask_shapes)} mask annotations")
            print(f"✅ Annotations are stored with type='mask'")

            # Check label types
            label_ids = set(s.get("label_id") for s in mask_shapes)
            print(f"✅ Using label IDs: {label_ids}")

            # Verify label types if we have label_map
            if label_map:
                all_mask_labels = True
                for label_id in label_ids:
                    if label_id in label_map:
                        label_type = label_map[label_id].get('type')
                        label_name = label_map[label_id].get('name')
                        if label_type != 'mask':
                            print(f"❌ Label {label_id} ({label_name}) has type='{label_type}', not 'mask'!")
                            all_mask_labels = False
                        else:
                            print(f"✅ Label {label_id} ({label_name}) has type='mask'")

                if all_mask_labels:
                    print("\n✅ ALL CHECKS PASSED - Annotations are correctly stored!")
                    print("\n⚠️  If masks still show as boxes in CVAT UI:")
                    print("  1. Check CVAT appearance settings: Enable 'Show bitmap'")
                    print("  2. Refresh the browser page")
                    print("  3. Check browser console for JavaScript errors")
                    print("  4. Verify the RLE format matches frontend expectations")
                    print("     (Frontend expects: [run_lengths..., left, top, right, bottom])")
                    print("     (Where right/bottom are INCLUSIVE end coordinates)")
                else:
                    print("\n❌ LABEL TYPE MISMATCH - This will cause box rendering!")
            else:
                print("⚠️  Could not verify label types")
        else:
            print("❌ No mask annotations found!")

        print("\nIf masks are showing as boxes, check:")
        print("  1. Task labels have type='mask' (not 'rectangle')")
        print("  2. Function metadata labels have type='mask'")
        print("  3. Annotations have type='mask' (not 'rectangle')")
        print("  4. Mask RLE format is valid (>= 6 values, valid bbox)")
        print("  5. CVAT appearance settings: 'Show bitmap' is enabled")

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

