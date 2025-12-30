#!/usr/bin/env python3
"""
Generate CVAT-compatible skeleton configurations for MediaPipe models.

This script generates skeleton label configurations that can be:
1. Used in function.yaml annotations for serverless functions
2. Pasted into CVAT's Raw label editor
3. Used to create projects/tasks with proper skeleton labels

The generated configs use the correct format:
- SVG with unescaped < and > (for innerHTML parsing)
- data-label-name (not data-label-id) for import
- data-element-id, data-node-id, and data-type attributes
"""

import json
from typing import List, Dict, Any, Tuple, Union
from pathlib import Path


def generate_hands_skeleton() -> Dict[str, Any]:
    """Generate hands skeleton configuration (42 keypoints: 21 per hand)."""

    # Hand keypoint names (21 per hand)
    hand_keypoints = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]

    # Create sublabels
    sublabels = []
    for idx, kp_name in enumerate(hand_keypoints):
        # Left hand
        sublabels.append({
            "id": idx,
            "name": f"left_{kp_name}",
            "type": "points",
            "attributes": []
        })
        # Right hand
        sublabels.append({
            "id": idx + 21,
            "name": f"right_{kp_name}",
            "type": "points",
            "attributes": []
        })

    # Generate SVG with proper coordinates and connections
    # Left hand positioned on left, right hand on right
    svg_parts = []

    # Helper to create circle element
    # data-node-id must be numeric (sublabel ID) to match data-node-from/to in edges
    def create_circle(cx: float, cy: float, element_id: int, node_name: str, node_id: int) -> str:
        return (
            f'<circle r="1" cx="{cx}" cy="{cy}" '
            f'data-type="element node" '
            f'data-element-id="{element_id}" '
            f'data-label-name="{node_name}" '
            f'data-node-id="{node_id}"></circle>'
        )

    # Helper to create line/edge element
    # CVAT requires numeric IDs in data-node-from and data-node-to (not names)
    def create_edge(x1: float, y1: float, x2: float, y2: float,
                    node_from: Union[str, int], node_to: Union[str, int]) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'data-type="edge" '
            f'data-node-from="{node_from}" '
            f'data-node-to="{node_to}"></line>'
        )

    # Left hand coordinates (centered around x=20, y=50)
    left_hand_positions = {
        "left_wrist": (20, 50),
        "left_thumb_cmc": (15, 45),
        "left_thumb_mcp": (12, 40),
        "left_thumb_ip": (10, 35),
        "left_thumb_tip": (8, 30),
        "left_index_mcp": (18, 42),
        "left_index_pip": (16, 37),
        "left_index_dip": (14, 32),
        "left_index_tip": (12, 27),
        "left_middle_mcp": (22, 40),
        "left_middle_pip": (20, 35),
        "left_middle_dip": (18, 30),
        "left_middle_tip": (16, 25),
        "left_ring_mcp": (26, 38),
        "left_ring_pip": (24, 33),
        "left_ring_dip": (22, 28),
        "left_ring_tip": (20, 23),
        "left_pinky_mcp": (30, 36),
        "left_pinky_pip": (28, 31),
        "left_pinky_dip": (26, 26),
        "left_pinky_tip": (24, 21),
    }

    # Right hand coordinates (centered around x=80, y=50)
    right_hand_positions = {
        "right_wrist": (80, 50),
        "right_thumb_cmc": (85, 45),
        "right_thumb_mcp": (88, 40),
        "right_thumb_ip": (90, 35),
        "right_thumb_tip": (92, 30),
        "right_index_mcp": (82, 42),
        "right_index_pip": (84, 37),
        "right_index_dip": (86, 32),
        "right_index_tip": (88, 27),
        "right_middle_mcp": (78, 40),
        "right_middle_pip": (80, 35),
        "right_middle_dip": (82, 30),
        "right_middle_tip": (84, 25),
        "right_ring_mcp": (74, 38),
        "right_ring_pip": (76, 33),
        "right_ring_dip": (78, 28),
        "right_ring_tip": (80, 23),
        "right_pinky_mcp": (70, 36),
        "right_pinky_pip": (72, 31),
        "right_pinky_dip": (74, 26),
        "right_pinky_tip": (76, 21),
    }

    # Create edges for left hand
    left_edges = [
        ("left_wrist", "left_thumb_cmc"),
        ("left_thumb_cmc", "left_thumb_mcp"),
        ("left_thumb_mcp", "left_thumb_ip"),
        ("left_thumb_ip", "left_thumb_tip"),
        ("left_wrist", "left_index_mcp"),
        ("left_index_mcp", "left_index_pip"),
        ("left_index_pip", "left_index_dip"),
        ("left_index_dip", "left_index_tip"),
        ("left_wrist", "left_middle_mcp"),
        ("left_middle_mcp", "left_middle_pip"),
        ("left_middle_pip", "left_middle_dip"),
        ("left_middle_dip", "left_middle_tip"),
        ("left_wrist", "left_ring_mcp"),
        ("left_ring_mcp", "left_ring_pip"),
        ("left_ring_pip", "left_ring_dip"),
        ("left_ring_dip", "left_ring_tip"),
        ("left_wrist", "left_pinky_mcp"),
        ("left_pinky_mcp", "left_pinky_pip"),
        ("left_pinky_pip", "left_pinky_dip"),
        ("left_pinky_dip", "left_pinky_tip"),
    ]

    # Create edges for right hand
    right_edges = [
        ("right_wrist", "right_thumb_cmc"),
        ("right_thumb_cmc", "right_thumb_mcp"),
        ("right_thumb_mcp", "right_thumb_ip"),
        ("right_thumb_ip", "right_thumb_tip"),
        ("right_wrist", "right_index_mcp"),
        ("right_index_mcp", "right_index_pip"),
        ("right_index_pip", "right_index_dip"),
        ("right_index_dip", "right_index_tip"),
        ("right_wrist", "right_middle_mcp"),
        ("right_middle_mcp", "right_middle_pip"),
        ("right_middle_pip", "right_middle_dip"),
        ("right_middle_dip", "right_middle_tip"),
        ("right_wrist", "right_ring_mcp"),
        ("right_ring_mcp", "right_ring_pip"),
        ("right_ring_pip", "right_ring_dip"),
        ("right_ring_dip", "right_ring_tip"),
        ("right_wrist", "right_pinky_mcp"),
        ("right_pinky_mcp", "right_pinky_pip"),
        ("right_pinky_pip", "right_pinky_dip"),
        ("right_pinky_dip", "right_pinky_tip"),
    ]

    # Create mapping from sublabel name to ID for edges
    # CVAT requires numeric IDs in data-node-from and data-node-to
    name_to_id = {sublabel["name"]: sublabel["id"] for sublabel in sublabels}

    # Build SVG: edges first, then circles (using numeric IDs)
    for from_node, to_node in left_edges:
        x1, y1 = left_hand_positions[from_node]
        x2, y2 = left_hand_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    for from_node, to_node in right_edges:
        x1, y1 = right_hand_positions[from_node]
        x2, y2 = right_hand_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add circles for all keypoints (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(left_hand_positions.items()):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    for element_id, (node_name, (cx, cy)) in enumerate(right_hand_positions.items(), start=21):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    svg = ''.join(svg_parts)

    return {
        "name": "hands-skeleton",
        "type": "skeleton",
        "attributes": [],
        "svg": svg,
        "sublabels": sublabels
    }


def generate_hands_shoulders_skeleton() -> Dict[str, Any]:
    """Generate hands-up-to-shoulders skeleton configuration (46 keypoints: shoulders, elbows, wrists, and hands)."""

    # Upper body keypoints (6: shoulders, elbows, wrists)
    upper_body_keypoints = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    ]

    # Hand keypoints (20 per hand, excluding wrist which is already in upper body)
    hand_keypoints = [
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]

    # Create sublabels: upper body first, then hands
    sublabels = []

    # Upper body keypoints (0-5)
    for idx, kp_name in enumerate(upper_body_keypoints):
        sublabels.append({
            "id": idx,
            "name": kp_name,
            "type": "points",
            "attributes": []
        })

    # Left hand keypoints (6-25, 20 keypoints excluding wrist)
    for idx, kp_name in enumerate(hand_keypoints):
        sublabels.append({
            "id": idx + 6,
            "name": f"left_{kp_name}",
            "type": "points",
            "attributes": []
        })

    # Right hand keypoints (26-45, 20 keypoints excluding wrist)
    for idx, kp_name in enumerate(hand_keypoints):
        sublabels.append({
            "id": idx + 26,
            "name": f"right_{kp_name}",
            "type": "points",
            "attributes": []
        })

    # Helper functions
    # data-node-id must be numeric (sublabel ID) to match data-node-from/to in edges
    def create_circle(cx: float, cy: float, element_id: int, node_name: str, node_id: int) -> str:
        return (
            f'<circle r="1" cx="{cx}" cy="{cy}" '
            f'data-type="element node" '
            f'data-element-id="{element_id}" '
            f'data-label-name="{node_name}" '
            f'data-node-id="{node_id}"></circle>'
        )

    # CVAT requires numeric IDs in data-node-from and data-node-to (not names)
    def create_edge(x1: float, y1: float, x2: float, y2: float,
                    node_from: Union[str, int], node_to: Union[str, int]) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'data-type="edge" '
            f'data-node-from="{node_from}" '
            f'data-node-to="{node_to}"></line>'
        )

    # Upper body keypoint positions
    upper_body_positions = {
        "left_shoulder": (30, 40),
        "right_shoulder": (70, 40),
        "left_elbow": (35, 70),
        "right_elbow": (65, 70),
        "left_wrist": (15, 80),
        "right_wrist": (85, 80),
    }

    # Left hand positions (connected to left_wrist at 15, 80)
    # Note: wrist is NOT included here - it's already in upper_body_positions
    left_hand_positions = {
        "left_thumb_cmc": (10, 75),
        "left_thumb_mcp": (8, 70),
        "left_thumb_ip": (6, 65),
        "left_thumb_tip": (4, 60),
        "left_index_mcp": (12, 72),
        "left_index_pip": (14, 67),
        "left_index_dip": (16, 62),
        "left_index_tip": (18, 57),
        "left_middle_mcp": (20, 52),
        "left_middle_pip": (22, 47),
        "left_middle_dip": (24, 42),
        "left_middle_tip": (26, 37),
        "left_ring_mcp": (28, 32),
        "left_ring_pip": (30, 27),
        "left_ring_dip": (32, 22),
        "left_ring_tip": (34, 17),
        "left_pinky_mcp": (36, 12),
        "left_pinky_pip": (38, 7),
        "left_pinky_dip": (40, 2),
        "left_pinky_tip": (42, -3),
    }

    # Right hand positions (connected to right_wrist at 85, 80)
    # Note: wrist is NOT included here - it's already in upper_body_positions
    right_hand_positions = {
        "right_thumb_cmc": (90, 75),
        "right_thumb_mcp": (92, 70),
        "right_thumb_ip": (94, 65),
        "right_thumb_tip": (96, 60),
        "right_index_mcp": (88, 72),
        "right_index_pip": (86, 67),
        "right_index_dip": (84, 62),
        "right_index_tip": (82, 57),
        "right_middle_mcp": (80, 52),
        "right_middle_pip": (78, 47),
        "right_middle_dip": (76, 42),
        "right_middle_tip": (74, 37),
        "right_ring_mcp": (72, 32),
        "right_ring_pip": (70, 27),
        "right_ring_dip": (68, 22),
        "right_ring_tip": (66, 17),
        "right_pinky_mcp": (64, 12),
        "right_pinky_pip": (62, 7),
        "right_pinky_dip": (60, 2),
        "right_pinky_tip": (58, -3),
    }

    # Upper body edges
    upper_body_edges = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
    ]

    # Left hand edges
    left_hand_edges = [
        ("left_wrist", "left_thumb_cmc"),
        ("left_thumb_cmc", "left_thumb_mcp"),
        ("left_thumb_mcp", "left_thumb_ip"),
        ("left_thumb_ip", "left_thumb_tip"),
        ("left_wrist", "left_index_mcp"),
        ("left_index_mcp", "left_index_pip"),
        ("left_index_pip", "left_index_dip"),
        ("left_index_dip", "left_index_tip"),
        ("left_wrist", "left_middle_mcp"),
        ("left_middle_mcp", "left_middle_pip"),
        ("left_middle_pip", "left_middle_dip"),
        ("left_middle_dip", "left_middle_tip"),
        ("left_wrist", "left_ring_mcp"),
        ("left_ring_mcp", "left_ring_pip"),
        ("left_ring_pip", "left_ring_dip"),
        ("left_ring_dip", "left_ring_tip"),
        ("left_wrist", "left_pinky_mcp"),
        ("left_pinky_mcp", "left_pinky_pip"),
        ("left_pinky_pip", "left_pinky_dip"),
        ("left_pinky_dip", "left_pinky_tip"),
    ]

    # Right hand edges
    right_hand_edges = [
        ("right_wrist", "right_thumb_cmc"),
        ("right_thumb_cmc", "right_thumb_mcp"),
        ("right_thumb_mcp", "right_thumb_ip"),
        ("right_thumb_ip", "right_thumb_tip"),
        ("right_wrist", "right_index_mcp"),
        ("right_index_mcp", "right_index_pip"),
        ("right_index_pip", "right_index_dip"),
        ("right_index_dip", "right_index_tip"),
        ("right_wrist", "right_middle_mcp"),
        ("right_middle_mcp", "right_middle_pip"),
        ("right_middle_pip", "right_middle_dip"),
        ("right_middle_dip", "right_middle_tip"),
        ("right_wrist", "right_ring_mcp"),
        ("right_ring_mcp", "right_ring_pip"),
        ("right_ring_pip", "right_ring_dip"),
        ("right_ring_dip", "right_ring_tip"),
        ("right_wrist", "right_pinky_mcp"),
        ("right_pinky_mcp", "right_pinky_pip"),
        ("right_pinky_pip", "right_pinky_dip"),
        ("right_pinky_dip", "right_pinky_tip"),
    ]

    # Create mapping from sublabel name to ID for edges
    # CVAT requires numeric IDs in data-node-from and data-node-to
    name_to_id = {sublabel["name"]: sublabel["id"] for sublabel in sublabels}

    svg_parts = []

    # Add upper body edges (using numeric IDs)
    for from_node, to_node in upper_body_edges:
        x1, y1 = upper_body_positions[from_node]
        x2, y2 = upper_body_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add left hand edges (using numeric IDs)
    for from_node, to_node in left_hand_edges:
        # Wrist is in upper_body_positions, not hand_positions
        x1, y1 = upper_body_positions.get(from_node, left_hand_positions.get(from_node))
        x2, y2 = upper_body_positions.get(to_node, left_hand_positions.get(to_node))
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add right hand edges (using numeric IDs)
    for from_node, to_node in right_hand_edges:
        # Wrist is in upper_body_positions, not hand_positions
        x1, y1 = upper_body_positions.get(from_node, right_hand_positions.get(from_node))
        x2, y2 = upper_body_positions.get(to_node, right_hand_positions.get(to_node))
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add circles for upper body keypoints (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(upper_body_positions.items()):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    # Add circles for left hand (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(left_hand_positions.items(), start=6):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    # Add circles for right hand (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(right_hand_positions.items(), start=27):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    svg = ''.join(svg_parts)

    return {
        "name": "hands-shoulders-skeleton",
        "type": "skeleton",
        "attributes": [],
        "svg": svg,
        "sublabels": sublabels
    }


def generate_person_skeleton() -> Dict[str, Any]:
    """Generate person skeleton configuration (17 body + 42 hand keypoints = 57 total)."""

    # Body keypoints (17)
    body_keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ]

    # Hand keypoints (20 per hand, excluding wrist which is already in body)
    # Note: wrist is already included in body_keypoints, so we skip it here to avoid duplicates
    hand_keypoints = [
        "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
        "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
        "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]

    # Create sublabels: body first, then hands
    sublabels = []

    # Body keypoints (0-16)
    for idx, kp_name in enumerate(body_keypoints):
        sublabels.append({
            "id": idx,
            "name": kp_name,
            "type": "points",
            "attributes": []
        })

    # Left hand keypoints (17-36, 20 keypoints excluding wrist)
    for idx, kp_name in enumerate(hand_keypoints):
        sublabels.append({
            "id": idx + 17,
            "name": f"left_{kp_name}",
            "type": "points",
            "attributes": []
        })

    # Right hand keypoints (37-56, 20 keypoints excluding wrist)
    for idx, kp_name in enumerate(hand_keypoints):
        sublabels.append({
            "id": idx + 37,
            "name": f"right_{kp_name}",
            "type": "points",
            "attributes": []
        })

    # Helper functions
    # data-node-id must be numeric (sublabel ID) to match data-node-from/to in edges
    def create_circle(cx: float, cy: float, element_id: int, node_name: str, node_id: int) -> str:
        return (
            f'<circle r="1" cx="{cx}" cy="{cy}" '
            f'data-type="element node" '
            f'data-element-id="{element_id}" '
            f'data-label-name="{node_name}" '
            f'data-node-id="{node_id}"></circle>'
        )

    # CVAT requires numeric IDs in data-node-from and data-node-to (not names)
    def create_edge(x1: float, y1: float, x2: float, y2: float,
                    node_from: Union[str, int], node_to: Union[str, int]) -> str:
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'data-type="edge" '
            f'data-node-from="{node_from}" '
            f'data-node-to="{node_to}"></line>'
        )

    # Body keypoint positions
    body_positions = {
        "nose": (50, 10),
        "left_eye": (40, 15),
        "right_eye": (60, 15),
        "left_ear": (35, 20),
        "right_ear": (65, 20),
        "left_shoulder": (30, 40),
        "right_shoulder": (70, 40),
        "left_elbow": (35, 70),
        "right_elbow": (65, 70),
        "left_wrist": (15, 80),
        "right_wrist": (85, 80),
        "left_hip": (40, 90),
        "right_hip": (60, 90),
        "left_knee": (45, 110),
        "right_knee": (55, 110),
        "left_ankle": (50, 130),
        "right_ankle": (60, 130),
    }

    # Left hand positions (connected to left_wrist at 15, 80)
    left_hand_positions = {
        "left_wrist": (15, 80),
        "left_thumb_cmc": (10, 75),
        "left_thumb_mcp": (8, 70),
        "left_thumb_ip": (6, 65),
        "left_thumb_tip": (4, 60),
        "left_index_mcp": (12, 72),
        "left_index_pip": (14, 67),
        "left_index_dip": (16, 62),
        "left_index_tip": (18, 57),
        "left_middle_mcp": (20, 52),
        "left_middle_pip": (22, 47),
        "left_middle_dip": (24, 42),
        "left_middle_tip": (26, 37),
        "left_ring_mcp": (28, 32),
        "left_ring_pip": (30, 27),
        "left_ring_dip": (32, 22),
        "left_ring_tip": (34, 17),
        "left_pinky_mcp": (36, 12),
        "left_pinky_pip": (38, 7),
        "left_pinky_dip": (40, 2),
        "left_pinky_tip": (42, -3),
    }

    # Right hand positions (connected to right_wrist at 85, 80)
    right_hand_positions = {
        "right_wrist": (85, 80),
        "right_thumb_cmc": (90, 75),
        "right_thumb_mcp": (92, 70),
        "right_thumb_ip": (94, 65),
        "right_thumb_tip": (96, 60),
        "right_index_mcp": (88, 72),
        "right_index_pip": (86, 67),
        "right_index_dip": (84, 62),
        "right_index_tip": (82, 57),
        "right_middle_mcp": (80, 52),
        "right_middle_pip": (78, 47),
        "right_middle_dip": (76, 42),
        "right_middle_tip": (74, 37),
        "right_ring_mcp": (72, 32),
        "right_ring_pip": (70, 27),
        "right_ring_dip": (68, 22),
        "right_ring_tip": (66, 17),
        "right_pinky_mcp": (64, 12),
        "right_pinky_pip": (62, 7),
        "right_pinky_dip": (60, 2),
        "right_pinky_tip": (58, -3),
    }

    # Body edges
    body_edges = [
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
    ]

    # Left hand edges
    left_hand_edges = [
        ("left_wrist", "left_thumb_cmc"),
        ("left_thumb_cmc", "left_thumb_mcp"),
        ("left_thumb_mcp", "left_thumb_ip"),
        ("left_thumb_ip", "left_thumb_tip"),
        ("left_wrist", "left_index_mcp"),
        ("left_index_mcp", "left_index_pip"),
        ("left_index_pip", "left_index_dip"),
        ("left_index_dip", "left_index_tip"),
        ("left_wrist", "left_middle_mcp"),
        ("left_middle_mcp", "left_middle_pip"),
        ("left_middle_pip", "left_middle_dip"),
        ("left_middle_dip", "left_middle_tip"),
        ("left_wrist", "left_ring_mcp"),
        ("left_ring_mcp", "left_ring_pip"),
        ("left_ring_pip", "left_ring_dip"),
        ("left_ring_dip", "left_ring_tip"),
        ("left_wrist", "left_pinky_mcp"),
        ("left_pinky_mcp", "left_pinky_pip"),
        ("left_pinky_pip", "left_pinky_dip"),
        ("left_pinky_dip", "left_pinky_tip"),
    ]

    # Right hand edges
    right_hand_edges = [
        ("right_wrist", "right_thumb_cmc"),
        ("right_thumb_cmc", "right_thumb_mcp"),
        ("right_thumb_mcp", "right_thumb_ip"),
        ("right_thumb_ip", "right_thumb_tip"),
        ("right_wrist", "right_index_mcp"),
        ("right_index_mcp", "right_index_pip"),
        ("right_index_pip", "right_index_dip"),
        ("right_index_dip", "right_index_tip"),
        ("right_wrist", "right_middle_mcp"),
        ("right_middle_mcp", "right_middle_pip"),
        ("right_middle_pip", "right_middle_dip"),
        ("right_middle_dip", "right_middle_tip"),
        ("right_wrist", "right_ring_mcp"),
        ("right_ring_mcp", "right_ring_pip"),
        ("right_ring_pip", "right_ring_dip"),
        ("right_ring_dip", "right_ring_tip"),
        ("right_wrist", "right_pinky_mcp"),
        ("right_pinky_mcp", "right_pinky_pip"),
        ("right_pinky_pip", "right_pinky_dip"),
        ("right_pinky_dip", "right_pinky_tip"),
    ]

    # Create mapping from sublabel name to ID for edges
    # CVAT requires numeric IDs in data-node-from and data-node-to
    name_to_id = {sublabel["name"]: sublabel["id"] for sublabel in sublabels}

    svg_parts = []

    # Add body edges (using numeric IDs)
    for from_node, to_node in body_edges:
        x1, y1 = body_positions[from_node]
        x2, y2 = body_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add left hand edges (using numeric IDs)
    for from_node, to_node in left_hand_edges:
        x1, y1 = left_hand_positions[from_node]
        x2, y2 = left_hand_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add right hand edges (using numeric IDs)
    for from_node, to_node in right_hand_edges:
        x1, y1 = right_hand_positions[from_node]
        x2, y2 = right_hand_positions[to_node]
        from_id = name_to_id[from_node]
        to_id = name_to_id[to_node]
        svg_parts.append(create_edge(x1, y1, x2, y2, from_id, to_id))

    # Add circles for body keypoints (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(body_positions.items()):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    # Add circles for left hand (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(left_hand_positions.items(), start=17):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    # Add circles for right hand (using numeric IDs from name_to_id mapping)
    for element_id, (node_name, (cx, cy)) in enumerate(right_hand_positions.items(), start=38):
        node_id = name_to_id[node_name]
        svg_parts.append(create_circle(cx, cy, element_id, node_name, node_id))

    svg = ''.join(svg_parts)

    return {
        "name": "person-skeleton",
        "type": "skeleton",
        "attributes": [],
        "svg": svg,
        "sublabels": sublabels
    }


def generate_json_for_raw_editor(skeletons: List[Dict[str, Any]]) -> str:
    """
    Generate JSON string for pasting into CVAT Raw label editor.

    The SVG should have unescaped < and > characters (actual XML),
    but quotes in attribute values will be escaped by JSON.stringify.
    """
    return json.dumps(skeletons, indent=2, ensure_ascii=False)


def generate_json_for_function_yaml(skeletons: List[Dict[str, Any]]) -> str:
    """
    Generate JSON string for function.yaml annotations.

    This needs to be a single-line JSON string that can be embedded in YAML.
    The SVG should have unescaped < and > characters.
    """
    # Convert to single-line JSON
    json_str = json.dumps(skeletons, ensure_ascii=False)
    # Escape quotes for YAML string literal
    json_str = json_str.replace('"', '\\"')
    return json_str


def main():
    """Generate skeleton configurations."""
    print("Generating MediaPipe skeleton configurations...")

    # Generate skeletons
    hands_skeleton = generate_hands_skeleton()
    hands_shoulders_skeleton = generate_hands_shoulders_skeleton()
    person_skeleton = generate_person_skeleton()

    skeletons = [hands_skeleton, hands_shoulders_skeleton, person_skeleton]

    # Generate JSON for raw editor (pretty-printed, unescaped SVG)
    raw_editor_json = generate_json_for_raw_editor(skeletons)

    # Save individual skeleton files (for reference/validation)
    output_dir = Path(__file__).parent

    # Save hands skeleton
    hands_file = output_dir / "mediapipe-hands-skeleton.json"
    with open(hands_file, 'w', encoding='utf-8') as f:
        json.dump([hands_skeleton], f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {hands_file}")

    # Save hands-shoulders skeleton
    hands_shoulders_file = output_dir / "mediapipe-hands-shoulders-skeleton.json"
    with open(hands_shoulders_file, 'w', encoding='utf-8') as f:
        json.dump([hands_shoulders_skeleton], f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {hands_shoulders_file}")

    # Save person skeleton
    person_file = output_dir / "mediapipe-person-skeleton.json"
    with open(person_file, 'w', encoding='utf-8') as f:
        json.dump([person_skeleton], f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {person_file}")

    # Save combined config for raw editor
    combined_file = output_dir / "mediapipe-skeletons-raw-editor.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(raw_editor_json)
    print(f"✓ Generated {combined_file}")
    print("\n" + "="*70)
    print("CONFIGURATION FOR CVAT RAW EDITOR:")
    print("="*70)
    print("Copy the contents of 'mediapipe-skeletons-raw-editor.json'")
    print("and paste it into CVAT's Raw label editor.")
    print("\nThe SVG format is correct:")
    print("  - Uses actual < and > characters (not HTML-escaped)")
    print("  - Uses data-label-name (CVAT will replace with data-label-id)")
    print("  - Has data-element-id, data-node-id, and data-type attributes")
    print("="*70)

    # Generate spec for function.yaml
    function_spec = generate_json_for_function_yaml([person_skeleton])
    spec_file = output_dir / "function-spec.txt"
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(function_spec)
    print(f"\n✓ Generated {spec_file}")
    print("This can be used in function.yaml annotations.spec field")
    print("="*70)


if __name__ == "__main__":
    main()

