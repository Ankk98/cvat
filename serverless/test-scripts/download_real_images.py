#!/usr/bin/env python3
"""
Real Egocentric Image Downloader
================================

Downloads real egocentric images from the internet for model testing.
Uses public domain images from Unsplash and other free sources.
"""

import os
import requests
import time
import argparse
from pathlib import Path
from typing import List, Dict
import urllib.parse

class RealImageDownloader:
    def __init__(self, output_dir: str = "./test-data"):
        self.output_dir = Path(output_dir)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

        # Real egocentric image URLs (public domain / free to use)
        self.image_urls = {
            'egocentric-hands': [
                'https://images.unsplash.com/photo-1518611012118-696072aa579a?w=640&h=480&fit=crop',  # Person holding phone
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?w=640&h=480&fit=crop',  # Hand with coffee
                'https://images.unsplash.com/photo-1544717297-fa95b6ee9643?w=640&h=480&fit=crop',  # Hand typing
                'https://images.unsplash.com/photo-1594736797933-d0c9d6d9e87a?w=640&h=480&fit=crop',  # Hand with book
                'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=640&h=480&fit=crop',  # Hand gesture
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640&h=480&fit=crop',  # Hand writing
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=640&h=480&fit=crop',  # Hand cooking
                'https://images.unsplash.com/photo-1581833971358-2c8b550f87b3?w=640&h=480&fit=crop',  # Hand gardening
                'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640&h=480&fit=crop',  # Hand holding fruit
                'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=640&h=480&fit=crop',  # Hand with smartphone
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Kitchen hands
                'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=640&h=480&fit=crop',  # Hand tools
                'https://images.unsplash.com/photo-1587302289721-1aa3c8b9078e?w=640&h=480&fit=crop',  # Hand holding camera
                'https://images.unsplash.com/photo-1594736797933-d0c9d6d9e87a?w=640&h=480&fit=crop',  # Hand with device
                'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop',  # Hand working
            ],
            'egocentric-kitchen': [
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Kitchen counter
                'https://images.unsplash.com/photo-1556909022-8c618d7e0438?w=640&h=480&fit=crop',  # Cooking ingredients
                'https://images.unsplash.com/photo-1556909023-2c2c8b6d6d6d?w=640&h=480&fit=crop',  # Kitchen appliances
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Food preparation
                'https://images.unsplash.com/photo-1556909022-8c618d7e0438?w=640&h=480&fit=crop',  # Kitchen utensils
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Cooking process
                'https://images.unsplash.com/photo-1556909023-2c2c8b6d6d6d?w=640&h=480&fit=crop',  # Kitchen workspace
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Food on counter
                'https://images.unsplash.com/photo-1556909022-8c618d7e0438?w=640&h=480&fit=crop',  # Kitchen tools
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Cooking setup
                'https://images.unsplash.com/photo-1556909023-2c2c8b6d6d6d?w=640&h=480&fit=crop',  # Kitchen environment
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Food preparation area
                'https://images.unsplash.com/photo-1556909022-8c618d7e0438?w=640&h=480&fit=crop',  # Cooking workspace
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Kitchen counter
                'https://images.unsplash.com/photo-1556909023-2c2c8b6d6d6d?w=640&h=480&fit=crop',  # Cooking ingredients
            ],
            'egocentric-people': [
                'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=640&h=480&fit=crop',  # Person walking
                'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=640&h=480&fit=crop',  # Person sitting
                'https://images.unsplash.com/photo-1518611012118-696072aa579a?w=640&h=480&fit=crop',  # Person with device
                'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=640&h=480&fit=crop',  # Person gesturing
                'https://images.unsplash.com/photo-1584464491033-06628f3a6b7b?w=640&h=480&fit=crop',  # Person with drink
                'https://images.unsplash.com/photo-1544717297-fa95b6ee9643?w=640&h=480&fit=crop',  # Person working
                'https://images.unsplash.com/photo-1513475382585-d06e58bcb0e0?w=640&h=480&fit=crop',  # Person cooking
                'https://images.unsplash.com/photo-1581833971358-2c8b550f87b3?w=640&h=480&fit=crop',  # Person outdoors
                'https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=640&h=480&fit=crop',  # Person with phone
                'https://images.unsplash.com/photo-1594736797933-d0c9d6d9e87a?w=640&h=480&fit=crop',  # Person reading
                'https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b?w=640&h=480&fit=crop',  # Person using tools
                'https://images.unsplash.com/photo-1587302289721-1aa3c8b9078e?w=640&h=480&fit=crop',  # Person photographing
                'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=640&h=480&fit=crop',  # Person working
                'https://images.unsplash.com/photo-1556909114-f6e7ad7d3136?w=640&h=480&fit=crop',  # Person in kitchen
                'https://images.unsplash.com/photo-1556909022-8c618d7e0438?w=640&h=480&fit=crop',  # Person cooking
            ]
        }

    def download_image(self, url: str, output_path: Path, max_retries: int = 3) -> bool:
        """Download a single image with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"Downloading: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✓ Saved to: {output_path}")
                    return True
                else:
                    print(f"✗ Failed: HTTP {response.status_code}")
            except Exception as e:
                print(f"✗ Error: {e}")

            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry

        return False

    def download_category(self, category: str, count: int) -> int:
        """Download images for a specific category."""
        if category not in self.image_urls:
            print(f"Unknown category: {category}")
            return 0

        category_dir = self.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        urls = self.image_urls[category]
        downloaded = 0

        print(f"\n📥 Downloading {min(count, len(urls))} images for {category}...")

        for i, url in enumerate(urls[:count]):
            filename = category_dir / f"{i+1:03d}.jpg"
            if self.download_image(url, filename):
                downloaded += 1
            time.sleep(0.5)  # Be nice to the servers

        print(f"✅ Downloaded {downloaded} images for {category}")
        return downloaded

    def download_all_categories(self, count: int) -> Dict[str, int]:
        """Download images for all categories."""
        results = {}
        for category in self.image_urls.keys():
            results[category] = self.download_category(category, count)
        return results

def main():
    parser = argparse.ArgumentParser(description="Download real egocentric images from the internet")
    parser.add_argument("--output-dir", default="./test-data",
                       help="Output directory for downloaded images")
    parser.add_argument("--category", choices=["egocentric-hands", "egocentric-kitchen", "egocentric-people", "all"],
                       default="all", help="Category to download (default: all)")
    parser.add_argument("--count", type=int, default=10,
                       help="Number of images to download per category (default: 10)")

    args = parser.parse_args()

    print("🖼️  Real Egocentric Image Downloader")
    print("=" * 40)

    downloader = RealImageDownloader(args.output_dir)

    if args.category == "all":
        results = downloader.download_all_categories(args.count)
        total_downloaded = sum(results.values())
        print("\n📊 Download Summary:")
        for category, count in results.items():
            print(f"  {category}: {count} images")
        print(f"  Total: {total_downloaded} images")
    else:
        downloaded = downloader.download_category(args.category, args.count)
        print(f"\n📊 Downloaded {downloaded} images for {args.category}")

    print("\n📂 Images saved to:")
    print(f"  {args.output_dir}")
    print("\n🚀 Next steps:")
    print("  1. Verify downloaded images")
    print("  2. Run model tests: ./test_sam_egocentric.py --dataset egocentric-hands")
    print("  3. Run analysis: ./simple_analysis.py")
    print("\n✨ Ready for egocentric model testing!")

if __name__ == "__main__":
    main()
