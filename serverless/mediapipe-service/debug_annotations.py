#!/usr/bin/env python3
"""
Debug script to analyze MediaPipe annotations vs CVAT annotations for egocentric videos.

This script:
1. Downloads frames from CVAT task/job
2. Sends them to MediaPipe service
3. Compares MediaPipe output with CVAT annotations
4. Identifies coordinate transformation issues
5. Analyzes detection rate and accuracy
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import cv2
import numpy as np
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AnnotationDebugger:
    """Debug MediaPipe annotations against CVAT annotations."""

    def __init__(self, cvat_url: str, task_id: int, job_id: Optional[int] = None,
                 mediapipe_url: str = "http://localhost:8000",
                 output_dir: str = "debug_output",
                 username: Optional[str] = None,
                 password: Optional[str] = None,
                 access_token: Optional[str] = None):
        self.cvat_url = cvat_url.rstrip('/')
        self.task_id = task_id
        self.job_id = job_id
        self.mediapipe_url = mediapipe_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Authentication
        self.session = requests.Session()
        self._setup_authentication(username, password, access_token)

        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_cvat_annotations': 0,
            'frames_with_mediapipe_detections': 0,
            'frames_with_both': 0,
            'coordinate_errors': [],
            'missing_detections': [],
            'false_detections': []
        }

    def _setup_authentication(self, username: Optional[str], password: Optional[str],
                              access_token: Optional[str]):
        """Setup CVAT authentication."""
        if access_token:
            # Personal Access Token (PAT) authentication
            self.session.headers['Authorization'] = f'Token {access_token}'
            logger.info("Using Personal Access Token authentication")
        elif username and password:
            # Basic authentication - login to get session
            try:
                login_url = f"{self.cvat_url}/api/auth/login"
                login_data = {
                    'username': username,
                    'password': password
                }
                response = self.session.post(login_url, json=login_data, timeout=30)
                response.raise_for_status()

                # CVAT uses session cookies and CSRF token
                if 'csrftoken' in self.session.cookies:
                    csrf_token = self.session.cookies['csrftoken']
                    self.session.headers['X-CSRFToken'] = csrf_token

                logger.info("Successfully authenticated with username/password")
            except Exception as e:
                logger.error(f"Authentication failed: {e}")
                raise
        else:
            # Try to get credentials from environment
            env_username = os.getenv('CVAT_USERNAME')
            env_password = os.getenv('CVAT_PASSWORD')
            env_token = os.getenv('CVAT_ACCESS_TOKEN')

            if env_token:
                self.session.headers['Authorization'] = f'Token {env_token}'
                logger.info("Using Personal Access Token from environment")
            elif env_username and env_password:
                try:
                    login_url = f"{self.cvat_url}/api/auth/login"
                    login_data = {
                        'username': env_username,
                        'password': env_password
                    }
                    response = self.session.post(login_url, json=login_data, timeout=30)
                    response.raise_for_status()

                    if 'csrftoken' in self.session.cookies:
                        csrf_token = self.session.cookies['csrftoken']
                        self.session.headers['X-CSRFToken'] = csrf_token

                    logger.info("Successfully authenticated with credentials from environment")
                except Exception as e:
                    logger.error(f"Authentication failed: {e}")
                    raise
            else:
                logger.warning("No authentication provided. Some CVAT endpoints may require authentication.")
                logger.warning("Set CVAT_USERNAME/CVAT_PASSWORD or CVAT_ACCESS_TOKEN environment variables, or use --username/--password or --token flags")

    def get_cvat_frame(self, frame_number: int) -> Optional[bytes]:
        """Download a frame from CVAT."""
        try:
            # CVAT API endpoint format: /api/tasks/{id}/data?type=frame&quality={quality}&number={number}
            # For jobs: /api/jobs/{id}/data?type=frame&quality={quality}&number={number}
            # Note: For jobs, frame_number should be relative to job start_frame
            if self.job_id:
                url = f"{self.cvat_url}/api/jobs/{self.job_id}/data"
            else:
                url = f"{self.cvat_url}/api/tasks/{self.task_id}/data"

            params = {
                'type': 'frame',
                'number': frame_number,
                'quality': 'original'
            }

            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to get frame {frame_number} from CVAT: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            return None

    def get_cvat_annotations(self, frame_number: int) -> Optional[Dict]:
        """Get annotations for a frame from CVAT."""
        try:
            # Get annotations for the job or task
            if self.job_id:
                url = f"{self.cvat_url}/api/jobs/{self.job_id}/annotations"
            else:
                url = f"{self.cvat_url}/api/tasks/{self.task_id}/annotations"

            params = {'frame': frame_number}
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Filter annotations for this frame
            shapes = data.get('shapes', [])
            frame_shapes = [s for s in shapes if s.get('frame') == frame_number]

            return {
                'shapes': frame_shapes,
                'frame': frame_number
            }
        except Exception as e:
            logger.error(f"Failed to get annotations for frame {frame_number}: {e}")
            return None

    def call_mediapipe(self, image_bytes: bytes) -> Optional[List[Dict]]:
        """Call MediaPipe service with image."""
        try:
            # Encode image as base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

            payload = {
                'image': image_b64,
                'threshold': 0.3
            }

            response = requests.post(
                f"{self.mediapipe_url}/detect",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"MediaPipe service call failed: {e}")
            return None

    def extract_skeleton_keypoints(self, annotation: Dict) -> List[Dict]:
        """Extract keypoints from CVAT skeleton annotation."""
        keypoints = []
        if annotation.get('type') == 'skeleton':
            elements = annotation.get('elements', [])
            for element in elements:
                if element.get('type') == 'points':
                    points = element.get('points', [])
                    if len(points) >= 2:
                        keypoints.append({
                            'label': element.get('label', 'unknown'),
                            'x': points[0],
                            'y': points[1],
                            'outside': element.get('outside', False),
                            'confidence': self._get_confidence(element)
                        })
        return keypoints

    def _get_confidence(self, element: Dict) -> float:
        """Extract confidence from element attributes."""
        attrs = element.get('attributes', [])
        for attr in attrs:
            if attr.get('name') == 'confidence':
                try:
                    return float(attr.get('value', 0.0))
                except:
                    return 0.0
        return 1.0

    def compare_annotations(self, cvat_anno: Dict, mediapipe_result: List[Dict],
                          image_width: int, image_height: int) -> Dict:
        """Compare CVAT annotations with MediaPipe results."""
        comparison = {
            'frame': cvat_anno.get('frame'),
            'cvat_keypoints': [],
            'mediapipe_keypoints': [],
            'matches': [],
            'coordinate_errors': [],
            'missing_in_mediapipe': [],
            'extra_in_mediapipe': []
        }

        # Extract CVAT keypoints
        for shape in cvat_anno.get('shapes', []):
            if shape.get('type') == 'skeleton':
                keypoints = self.extract_skeleton_keypoints(shape)
                comparison['cvat_keypoints'].extend(keypoints)

        # Extract MediaPipe keypoints
        for skeleton in mediapipe_result:
            if skeleton.get('type') == 'skeleton':
                elements = skeleton.get('elements', [])
                for element in elements:
                    if element.get('type') == 'points':
                        points = element.get('points', [])
                        if len(points) >= 2:
                            comparison['mediapipe_keypoints'].append({
                                'label': element.get('label', 'unknown'),
                                'x': points[0],
                                'y': points[1],
                                'outside': element.get('outside', False),
                                'confidence': self._get_confidence(element)
                            })

        # Match keypoints by label
        cvat_labels = {kp['label']: kp for kp in comparison['cvat_keypoints']}
        mp_labels = {kp['label']: kp for kp in comparison['mediapipe_keypoints']}

        # Find matches and errors
        all_labels = set(cvat_labels.keys()) | set(mp_labels.keys())
        for label in all_labels:
            cvat_kp = cvat_labels.get(label)
            mp_kp = mp_labels.get(label)

            if cvat_kp and mp_kp:
                # Calculate distance
                dx = abs(cvat_kp['x'] - mp_kp['x'])
                dy = abs(cvat_kp['y'] - mp_kp['y'])
                distance = np.sqrt(dx**2 + dy**2)

                # Normalize by image size
                max_dim = max(image_width, image_height)
                normalized_distance = distance / max_dim if max_dim > 0 else 0

                comparison['matches'].append({
                    'label': label,
                    'cvat': cvat_kp,
                    'mediapipe': mp_kp,
                    'distance': distance,
                    'normalized_distance': normalized_distance
                })

                if normalized_distance > 0.1:  # 10% of image size threshold
                    comparison['coordinate_errors'].append({
                        'label': label,
                        'distance': distance,
                        'normalized_distance': normalized_distance,
                        'cvat_pos': (cvat_kp['x'], cvat_kp['y']),
                        'mp_pos': (mp_kp['x'], mp_kp['y'])
                    })
            elif cvat_kp and not mp_kp:
                comparison['missing_in_mediapipe'].append(cvat_kp)
            elif mp_kp and not cvat_kp:
                comparison['extra_in_mediapipe'].append(mp_kp)

        return comparison

    def visualize_comparison(self, image_bytes: bytes, comparison: Dict,
                            output_path: Path):
        """Visualize comparison on image."""
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Draw CVAT keypoints in green
        for kp in comparison['cvat_keypoints']:
            if not kp['outside']:
                x, y = int(kp['x']), int(kp['y'])
                cv2.circle(img_cv, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(img_cv, kp['label'], (x+10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # Draw MediaPipe keypoints in red
        for kp in comparison['mediapipe_keypoints']:
            if not kp['outside']:
                x, y = int(kp['x']), int(kp['y'])
                cv2.circle(img_cv, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(img_cv, kp['label'], (x+10, y+15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # Draw lines between matching keypoints
        for match in comparison['matches']:
            cvat_pos = (int(match['cvat']['x']), int(match['cvat']['y']))
            mp_pos = (int(match['mediapipe']['x']), int(match['mediapipe']['y']))
            color = (255, 255, 0) if match['normalized_distance'] > 0.1 else (255, 0, 255)
            cv2.line(img_cv, cvat_pos, mp_pos, color, 1)

        # Save visualization
        cv2.imwrite(str(output_path), img_cv)

    def analyze_frame_range(self, start_frame: int, end_frame: int,
                           step: int = 1, visualize: bool = True):
        """Analyze a range of frames."""
        logger.info(f"Analyzing frames {start_frame} to {end_frame} (step {step})")

        results = []

        for frame_num in range(start_frame, end_frame + 1, step):
            logger.info(f"Processing frame {frame_num}...")

            # Get frame from CVAT
            frame_bytes = self.get_cvat_frame(frame_num)
            if not frame_bytes:
                logger.warning(f"Could not get frame {frame_num}")
                continue

            # Get image dimensions
            image = Image.open(io.BytesIO(frame_bytes))
            image_width, image_height = image.size

            # Get CVAT annotations
            cvat_anno = self.get_cvat_annotations(frame_num)
            has_cvat = cvat_anno and len(cvat_anno.get('shapes', [])) > 0

            # Call MediaPipe
            mediapipe_result = self.call_mediapipe(frame_bytes)
            has_mp = mediapipe_result and len(mediapipe_result) > 0

            # Update statistics
            self.stats['total_frames'] += 1
            if has_cvat:
                self.stats['frames_with_cvat_annotations'] += 1
            if has_mp:
                self.stats['frames_with_mediapipe_detections'] += 1
            if has_cvat and has_mp:
                self.stats['frames_with_both'] += 1

            # Compare if both exist
            if has_cvat and has_mp:
                comparison = self.compare_annotations(
                    cvat_anno, mediapipe_result, image_width, image_height
                )
                results.append(comparison)

                # Visualize if requested
                if visualize:
                    vis_path = self.output_dir / f"frame_{frame_num:06d}_comparison.jpg"
                    self.visualize_comparison(frame_bytes, comparison, vis_path)

                # Log errors
                if comparison['coordinate_errors']:
                    self.stats['coordinate_errors'].extend(comparison['coordinate_errors'])
                if comparison['missing_in_mediapipe']:
                    self.stats['missing_detections'].append({
                        'frame': frame_num,
                        'missing': comparison['missing_in_mediapipe']
                    })
                if comparison['extra_in_mediapipe']:
                    self.stats['false_detections'].append({
                        'frame': frame_num,
                        'extra': comparison['extra_in_mediapipe']
                    })
            elif has_cvat and not has_mp:
                logger.warning(f"Frame {frame_num}: CVAT has annotations but MediaPipe detected nothing")
                self.stats['missing_detections'].append({
                    'frame': frame_num,
                    'missing': 'all_keypoints'
                })
            elif has_mp and not has_cvat:
                logger.info(f"Frame {frame_num}: MediaPipe detected but no CVAT annotations")

        # Save results
        results_path = self.output_dir / "analysis_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'statistics': self.stats,
                'detailed_results': results
            }, f, indent=2)

        logger.info(f"Analysis complete. Results saved to {results_path}")
        self.print_summary()

    def print_summary(self):
        """Print analysis summary."""
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Total frames analyzed: {self.stats['total_frames']}")
        print(f"Frames with CVAT annotations: {self.stats['frames_with_cvat_annotations']}")
        print(f"Frames with MediaPipe detections: {self.stats['frames_with_mediapipe_detections']}")
        print(f"Frames with both: {self.stats['frames_with_both']}")
        print(f"Coordinate errors: {len(self.stats['coordinate_errors'])}")
        print(f"Missing detections: {len(self.stats['missing_detections'])}")
        print(f"False detections: {len(self.stats['false_detections'])}")

        if self.stats['coordinate_errors']:
            avg_error = np.mean([e['normalized_distance'] for e in self.stats['coordinate_errors']])
            max_error = max([e['normalized_distance'] for e in self.stats['coordinate_errors']])
            print(f"\nAverage coordinate error: {avg_error:.2%} of image size")
            print(f"Maximum coordinate error: {max_error:.2%} of image size")

        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Debug MediaPipe annotations against CVAT annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Authentication Options:
  The script supports multiple authentication methods:

  1. Personal Access Token (recommended):
     --token YOUR_TOKEN
     or set CVAT_ACCESS_TOKEN environment variable

  2. Username/Password:
     --username USERNAME --password PASSWORD
     or set CVAT_USERNAME and CVAT_PASSWORD environment variables

  3. No authentication (may not work for protected endpoints):
     Script will attempt to access without authentication

Examples:
  # Using Personal Access Token
  python debug_annotations.py --task-id 14 --job-id 10 --token YOUR_TOKEN

  # Using username/password
  python debug_annotations.py --task-id 14 --job-id 10 --username admin --password admin

  # Using environment variables
  export CVAT_ACCESS_TOKEN=YOUR_TOKEN
  python debug_annotations.py --task-id 14 --job-id 10
        """
    )
    parser.add_argument('--cvat-url', default='http://localhost:8080',
                       help='CVAT server URL')
    parser.add_argument('--task-id', type=int, required=True,
                       help='CVAT task ID')
    parser.add_argument('--job-id', type=int, default=None,
                       help='CVAT job ID (optional)')
    parser.add_argument('--mediapipe-url', default='http://localhost:8000',
                       help='MediaPipe service URL')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Start frame number')
    parser.add_argument('--end-frame', type=int, default=100,
                       help='End frame number')
    parser.add_argument('--step', type=int, default=1,
                       help='Frame step')
    parser.add_argument('--output-dir', default='debug_output',
                       help='Output directory for results')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Skip visualization')

    # Authentication arguments
    parser.add_argument('--username', type=str, default=None,
                       help='CVAT username (alternative to token)')
    parser.add_argument('--password', type=str, default=None,
                       help='CVAT password (alternative to token)')
    parser.add_argument('--token', type=str, default=None,
                       help='CVAT Personal Access Token (recommended)')

    args = parser.parse_args()

    debugger = AnnotationDebugger(
        cvat_url=args.cvat_url,
        task_id=args.task_id,
        job_id=args.job_id,
        mediapipe_url=args.mediapipe_url,
        output_dir=args.output_dir,
        username=args.username,
        password=args.password,
        access_token=args.token
    )

    debugger.analyze_frame_range(
        args.start_frame,
        args.end_frame,
        args.step,
        visualize=not args.no_visualize
    )


if __name__ == '__main__':
    main()

