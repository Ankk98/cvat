#!/usr/bin/env python3
"""
Download Ground Truth Annotations for Egocentric Datasets
=========================================================

Downloads or generates ground truth annotations for egocentric test images.
Supports multiple annotation formats (COCO, CVAT XML, JSON).

Usage:
    python download_gt_annotations.py --dataset-path test-data/egocentric-hands --format coco
    python download_gt_annotations.py --dataset-path test-data/egocentric-hands --format cvat
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

class GTAnnotationDownloader:
    """Download or generate ground truth annotations for test datasets."""

    def __init__(self, dataset_path: Path, format: str = "json"):
        self.dataset_path = Path(dataset_path)
        self.format = format.lower()
        self.annotations = {}

    def load_images(self) -> List[Path]:
        """Load all images from dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in self.dataset_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        return sorted(images)

    def create_skeleton_annotation(self, image_path: Path, keypoints: Dict[str, List[float]]) -> Dict:
        """Create skeleton annotation in CVAT-compatible format."""
        img = Image.open(image_path)
        width, height = img.size

        elements = []
        for label, coords in keypoints.items():
            if len(coords) >= 2:
                elements.append({
                    "label": label,
                    "type": "points",
                    "points": coords[:2],  # x, y
                    "outside": False,
                    "occluded": False,
                    "attributes": [
                        {"name": "confidence", "value": "1.0"}
                    ]
                })

        return {
            "type": "skeleton",
            "label": "person",
            "elements": elements,
            "frame": 0  # For single image annotations
        }

    def create_mask_annotation(self, image_path: Path, mask: np.ndarray, label: str = "hand") -> Dict:
        """Create mask annotation in CVAT-compatible format."""
        # Encode mask as RLE (Run-Length Encoding)
        rle = self._encode_rle(mask)

        # Get bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
        else:
            x_min = y_min = x_max = y_max = 0

        return {
            "type": "mask",
            "label": label,
            "rle": rle,
            "left": int(x_min),
            "top": int(y_min),
            "width": int(x_max - x_min + 1),
            "height": int(y_max - y_min + 1),
            "frame": 0
        }

    def _encode_rle(self, mask: np.ndarray) -> List[int]:
        """Encode binary mask as RLE."""
        flat_mask = mask.flatten()
        rle = []
        i = 0
        while i < len(flat_mask):
            if flat_mask[i] == 1:
                start = i
                while i < len(flat_mask) and flat_mask[i] == 1:
                    i += 1
                length = i - start
                rle.extend([start, length])
            else:
                i += 1
        return rle

    def save_annotations_json(self, output_path: Path):
        """Save annotations in JSON format."""
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def save_annotations_cvat_xml(self, output_path: Path):
        """Save annotations in CVAT XML format."""
        root = ET.Element("annotations")
        version = ET.SubElement(root, "version")
        version.text = "1.1"

        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")

        for image_name, annotation_data in self.annotations.items():
            image_elem = ET.SubElement(root, "image")
            image_elem.set("name", image_name)
            image_elem.set("id", "0")

            # Add skeleton annotations
            if "skeletons" in annotation_data:
                for skeleton in annotation_data["skeletons"]:
                    skeleton_elem = ET.SubElement(image_elem, "skeleton")
                    skeleton_elem.set("label", skeleton.get("label", "person"))

                    for element in skeleton.get("elements", []):
                        points_elem = ET.SubElement(skeleton_elem, "points")
                        points_elem.set("label", element["label"])
                        points_elem.set("points", f"{element['points'][0]},{element['points'][1]}")
                        points_elem.set("outside", "0" if not element.get("outside", False) else "1")
                        points_elem.set("occluded", "0" if not element.get("occluded", False) else "1")

            # Add mask annotations
            if "masks" in annotation_data:
                for mask_data in annotation_data["masks"]:
                    mask_elem = ET.SubElement(image_elem, "mask")
                    mask_elem.set("label", mask_data.get("label", "hand"))
                    mask_elem.set("rle", ",".join(map(str, mask_data["rle"])))
                    mask_elem.set("left", str(mask_data["left"]))
                    mask_elem.set("top", str(mask_data["top"]))
                    mask_elem.set("width", str(mask_data["width"]))
                    mask_elem.set("height", str(mask_data["height"]))

        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    def generate_from_existing(self, annotation_file: Optional[Path] = None):
        """Generate annotations from existing file or create empty structure."""
        images = self.load_images()

        if annotation_file and annotation_file.exists():
            # Load existing annotations
            if annotation_file.suffix == '.json':
                with open(annotation_file, 'r') as f:
                    self.annotations = json.load(f)
            elif annotation_file.suffix == '.xml':
                # Parse CVAT XML
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                for image_elem in root.findall('image'):
                    image_name = image_elem.get('name')
                    self.annotations[image_name] = {
                        "skeletons": [],
                        "masks": []
                    }
        else:
            # Create empty annotation structure
            for image_path in images:
                self.annotations[image_path.name] = {
                    "skeletons": [],
                    "masks": [],
                    "image_path": str(image_path)
                }

    def save(self, output_path: Path):
        """Save annotations in specified format."""
        if self.format == "json":
            self.save_annotations_json(output_path)
        elif self.format == "cvat" or self.format == "xml":
            self.save_annotations_cvat_xml(output_path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")


def main():
    parser = argparse.ArgumentParser(
        description="Download or generate ground truth annotations for egocentric datasets"
    )
    parser.add_argument('--dataset-path', required=True,
                       help='Path to dataset directory with images')
    parser.add_argument('--format', choices=['json', 'cvat', 'xml'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--output', help='Output annotation file path')
    parser.add_argument('--input', help='Input annotation file to convert')

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return 1

    downloader = GTAnnotationDownloader(dataset_path, args.format)
    downloader.generate_from_existing(Path(args.input) if args.input else None)

    output_path = Path(args.output) if args.output else dataset_path / f"annotations.{args.format}"
    downloader.save(output_path)

    print(f"✅ Annotations saved to: {output_path}")
    print(f"   Format: {args.format}")
    print(f"   Images: {len(downloader.annotations)}")

    return 0


if __name__ == '__main__':
    exit(main())

