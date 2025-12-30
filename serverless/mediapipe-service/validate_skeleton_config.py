#!/usr/bin/env python3
"""
Skeleton Configuration Validator

Validates MediaPipe skeleton JSON configuration files for CVAT integration.
Checks:
- JSON structure validity
- SVG XML validity
- Node/edge consistency
- Sublabel matching
- Integration completeness
"""

import json
import sys
import xml.etree.ElementTree as ET
import html
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class SkeletonValidator:
    """Validates skeleton configuration files."""

    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.data = None
        self.svg_root = None
        self.errors = []
        self.warnings = []

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks.

        Returns:
            (is_valid, errors, warnings)
        """
        try:
            # Load and validate JSON
            self._load_json()
            self._validate_json_structure()

            # Load and validate SVG
            self._load_svg()
            self._validate_svg_structure()

            # Validate integration
            self._validate_node_consistency()
            self._validate_edge_consistency()
            self._validate_sublabel_matching()
            self._validate_completeness()

        except ValidationError as e:
            self.errors.append(str(e))
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _load_json(self):
        """Load and parse JSON file."""
        if not self.json_path.exists():
            raise ValidationError(f"File not found: {self.json_path}")

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}")

    def _validate_json_structure(self):
        """Validate JSON structure matches CVAT skeleton format."""
        if not isinstance(self.data, list):
            raise ValidationError("Root must be a list")

        if len(self.data) == 0:
            raise ValidationError("JSON must contain at least one skeleton definition")

        for idx, skeleton in enumerate(self.data):
            if not isinstance(skeleton, dict):
                raise ValidationError(f"Skeleton {idx}: must be an object")

            # Required fields
            required_fields = ['name', 'type', 'svg', 'sublabels']
            for field in required_fields:
                if field not in skeleton:
                    raise ValidationError(f"Skeleton {idx}: missing required field '{field}'")

            # Type validation
            if skeleton['type'] != 'skeleton':
                raise ValidationError(f"Skeleton {idx}: type must be 'skeleton'")

            if not isinstance(skeleton['sublabels'], list):
                raise ValidationError(f"Skeleton {idx}: sublabels must be a list")

            # Validate sublabels
            for sub_idx, sublabel in enumerate(skeleton['sublabels']):
                if not isinstance(sublabel, dict):
                    raise ValidationError(f"Skeleton {idx}, sublabel {sub_idx}: must be an object")

                required_sublabel_fields = ['id', 'name', 'type']
                for field in required_sublabel_fields:
                    if field not in sublabel:
                        raise ValidationError(
                            f"Skeleton {idx}, sublabel {sub_idx}: missing required field '{field}'"
                        )

                if sublabel['type'] != 'points':
                    raise ValidationError(
                        f"Skeleton {idx}, sublabel {sub_idx}: type must be 'points'"
                    )

    def _load_svg(self):
        """Load and parse SVG from JSON."""
        if not self.data:
            return

        skeleton = self.data[0]  # Validate first skeleton
        svg_string = skeleton.get('svg', '')

        if not svg_string:
            raise ValidationError("SVG string is empty")

        try:
            # Decode HTML entities (e.g., &quot; -> ")
            svg_string = html.unescape(svg_string)

            # Wrap in root element for parsing (SVG fragments)
            wrapped_svg = f"<root>{svg_string}</root>"
            self.svg_root = ET.fromstring(wrapped_svg)
        except ET.ParseError as e:
            raise ValidationError(f"Invalid SVG XML: {e}")

    def _validate_svg_structure(self):
        """Validate SVG structure."""
        if self.svg_root is None:
            return

        circles = self.svg_root.findall('.//circle')
        lines = self.svg_root.findall('.//line')

        if len(circles) == 0:
            raise ValidationError("SVG must contain at least one circle (keypoint)")

        # Validate circles have required attributes
        # CVAT import format uses data-label-name, stored format uses data-label-id
        for idx, circle in enumerate(circles):
            required_attrs = ['data-node-id']
            for attr in required_attrs:
                if attr not in circle.attrib:
                    raise ValidationError(
                        f"Circle {idx}: missing required attribute '{attr}'"
                    )

            # Must have either data-label-id OR data-label-name
            has_label_id = 'data-label-id' in circle.attrib
            has_label_name = 'data-label-name' in circle.attrib
            if not has_label_id and not has_label_name:
                raise ValidationError(
                    f"Circle {idx}: missing required attribute 'data-label-id' or 'data-label-name'"
                )

        # Validate lines (edges) have required attributes
        for idx, line in enumerate(lines):
            required_attrs = ['data-type', 'data-node-from', 'data-node-to']
            for attr in required_attrs:
                if attr not in line.attrib:
                    raise ValidationError(
                        f"Line {idx}: missing required attribute '{attr}'"
                    )

            if line.attrib.get('data-type') != 'edge':
                raise ValidationError(
                    f"Line {idx}: data-type must be 'edge'"
                )

    def _validate_node_consistency(self):
        """Validate that all nodes referenced in edges exist."""
        if self.svg_root is None:
            return

        # Get all node IDs from circles
        node_ids = set()
        for circle in self.svg_root.findall('.//circle'):
            node_id = circle.attrib.get('data-node-id')
            if node_id:
                node_ids.add(node_id)

        # Check all edges reference existing nodes
        for idx, line in enumerate(self.svg_root.findall('.//line')):
            node_from = line.attrib.get('data-node-from')
            node_to = line.attrib.get('data-node-to')

            if node_from and node_from not in node_ids:
                self.errors.append(
                    f"Line {idx}: data-node-from='{node_from}' references non-existent node"
                )

            if node_to and node_to not in node_ids:
                self.errors.append(
                    f"Line {idx}: data-node-to='{node_to}' references non-existent node"
                )

    def _validate_edge_consistency(self):
        """Validate edge consistency and detect issues."""
        if self.svg_root is None:
            return

        # Build node-to-edge mapping
        node_edges = defaultdict(list)
        edges = []

        for line in self.svg_root.findall('.//line'):
            node_from = line.attrib.get('data-node-from')
            node_to = line.attrib.get('data-node-to')

            if node_from and node_to:
                edges.append((node_from, node_to))
                node_edges[node_from].append(node_to)
                node_edges[node_to].append(node_from)

        # Check for duplicate edges
        edge_set = set(edges)
        if len(edge_set) != len(edges):
            self.warnings.append("Duplicate edges detected (same edge defined multiple times)")

        # Check for self-loops
        for node_from, node_to in edges:
            if node_from == node_to:
                self.warnings.append(f"Self-loop detected: {node_from} -> {node_to}")

        # Check for isolated nodes (nodes with no edges)
        node_ids = set()
        for circle in self.svg_root.findall('.//circle'):
            node_id = circle.attrib.get('data-node-id')
            if node_id:
                node_ids.add(node_id)

        isolated_nodes = node_ids - set(node_edges.keys())
        if isolated_nodes:
            self.warnings.append(
                f"Isolated nodes (no edges): {', '.join(sorted(isolated_nodes))}"
            )

    def _validate_sublabel_matching(self):
        """Validate that sublabels match SVG nodes."""
        if not self.data or self.svg_root is None:
            return

        skeleton = self.data[0]
        sublabels = skeleton.get('sublabels', [])

        # Build mapping: sublabel name -> sublabel ID
        sublabel_name_to_id = {}
        sublabel_id_to_name = {}

        for sublabel in sublabels:
            name = sublabel.get('name')
            sublabel_id = sublabel.get('id')

            if name:
                sublabel_name_to_id[name] = sublabel_id
            if sublabel_id is not None:
                sublabel_id_to_name[sublabel_id] = name

        # Check circles: data-node-id should match sublabel name
        # and data-label-id (if present) should match sublabel id
        # or data-label-name (if present) should match sublabel name
        for circle in self.svg_root.findall('.//circle'):
            node_id = circle.attrib.get('data-node-id')
            label_id = circle.attrib.get('data-label-id')
            label_name = circle.attrib.get('data-label-name')

            if node_id:
                # Check if node_id matches a sublabel name
                if node_id not in sublabel_name_to_id:
                    self.errors.append(
                        f"Circle with node-id='{node_id}': no matching sublabel with name '{node_id}'"
                    )
                else:
                    # If data-label-id is present, check it matches sublabel id
                    if label_id:
                        try:
                            label_id_int = int(label_id)
                            expected_id = sublabel_name_to_id[node_id]
                            if label_id_int != expected_id:
                                self.errors.append(
                                    f"Circle with node-id='{node_id}': data-label-id='{label_id}' "
                                    f"does not match sublabel id '{expected_id}'"
                                )
                        except (ValueError, TypeError):
                            self.errors.append(
                                f"Circle with node-id='{node_id}': data-label-id='{label_id}' is not a valid integer"
                            )

                    # If data-label-name is present, check it matches sublabel name
                    if label_name:
                        if label_name != node_id:
                            self.errors.append(
                                f"Circle with node-id='{node_id}': data-label-name='{label_name}' "
                                f"does not match sublabel name '{node_id}'"
                            )

    def _validate_completeness(self):
        """Validate completeness of skeleton structure."""
        if not self.data or self.svg_root is None:
            return

        skeleton = self.data[0]
        sublabels = skeleton.get('sublabels', [])

        # Count nodes and edges
        circles = self.svg_root.findall('.//circle')
        lines = self.svg_root.findall('.//line')

        # Check if we have expected structure
        if len(sublabels) != len(circles):
            self.warnings.append(
                f"Mismatch: {len(sublabels)} sublabels but {len(circles)} circles in SVG"
            )

        # For hand skeletons, check if we have edges for all fingers
        skeleton_name = skeleton.get('name', '').lower()
        if 'hand' in skeleton_name or skeleton_name == 'hands':
            # Each hand should have 21 keypoints and 20 edges
            expected_keypoints_per_hand = 21
            expected_edges_per_hand = 20

            if len(sublabels) == expected_keypoints_per_hand * 2:  # Two hands
                expected_edges = expected_edges_per_hand * 2
                if len(lines) != expected_edges:
                    self.warnings.append(
                        f"Hand skeleton: expected {expected_edges} edges (20 per hand), "
                        f"found {len(lines)}"
                    )

        # For person skeleton, check if we have body + hand edges
        if skeleton_name == 'person':
            # Should have body edges + hand edges
            # Body: ~16 edges, Hands: 20 edges each = 40, Total: ~56
            if len(lines) < 50:
                self.warnings.append(
                    f"Person skeleton: expected ~56 edges (16 body + 40 hands), found {len(lines)}"
                )

    def _get_structure_summary(self) -> Dict:
        """Get summary of skeleton structure."""
        if not self.data or self.svg_root is None:
            return {}

        skeleton = self.data[0]
        circles = self.svg_root.findall('.//circle')
        lines = self.svg_root.findall('.//line')
        sublabels = skeleton.get('sublabels', [])

        # Count nodes by type
        node_ids = [circle.attrib.get('data-node-id') for circle in circles]

        # Count edges
        edges = []
        for line in lines:
            node_from = line.attrib.get('data-node-from')
            node_to = line.attrib.get('data-node-to')
            if node_from and node_to:
                edges.append((node_from, node_to))

        return {
            'name': skeleton.get('name', 'unknown'),
            'sublabels_count': len(sublabels),
            'nodes_count': len(circles),
            'edges_count': len(lines),
            'unique_edges': len(set(edges)),
            'node_ids': node_ids
        }

    def print_report(self, detailed: bool = False):
        """Print validation report."""
        print(f"\n{'='*70}")
        print(f"Validation Report: {self.json_path.name}")
        print(f"{'='*70}\n")

        is_valid, errors, warnings = self.validate()

        if is_valid:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")

        # Print structure summary
        summary = self._get_structure_summary()
        if summary:
            print(f"\n📊 Structure Summary:")
            print(f"  Skeleton Name: {summary['name']}")
            print(f"  Sublabels: {summary['sublabels_count']}")
            print(f"  Nodes (circles): {summary['nodes_count']}")
            print(f"  Edges (lines): {summary['edges_count']}")
            print(f"  Unique Edges: {summary['unique_edges']}")

            if detailed and summary['node_ids']:
                print(f"\n  Node IDs: {', '.join(summary['node_ids'][:10])}")
                if len(summary['node_ids']) > 10:
                    print(f"    ... and {len(summary['node_ids']) - 10} more")

        if errors:
            print(f"\n❌ ERRORS ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")

        if not errors and not warnings:
            print("\n✅ No issues found!")
            print("✅ JSON structure is valid")
            print("✅ SVG structure is valid")
            print("✅ All nodes and edges are consistent")
            print("✅ Sublabels match SVG nodes")

        print(f"\n{'='*70}\n")

        return is_valid


def main():
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Validate MediaPipe skeleton JSON configuration files for CVAT'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='JSON files to validate'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed structure information'
    )

    args = parser.parse_args()

    all_valid = True

    for json_file in args.files:
        validator = SkeletonValidator(json_file)
        is_valid = validator.print_report(detailed=args.detailed)
        if not is_valid:
            all_valid = False

    if all_valid:
        print("✅ All files validated successfully!")
        sys.exit(0)
    else:
        print("❌ Some files failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()

