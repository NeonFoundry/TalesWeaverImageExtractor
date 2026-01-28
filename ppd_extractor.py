#!/usr/bin/env python3
"""
TalesWeaver PPD (Polygon Path Data) Extractor by NeonFoundry 2026

Extracts and visualizes polygon data from TalesWeaver Map PPD files.
PPD files contain polygon collision/path data for game maps.

PPD Format:
- Bytes 0-1: Version (uint16 LE, usually 1)
- Bytes 2-3: Polygon count (uint16 LE)
- Bytes 4-7: Unknown/padding
- Bytes 8+: Padding zeros until polygon data
- Polygon entries: point_count (uint16) followed by X,Y pairs (uint16 each)
"""

import struct
import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class Polygon:
    """Represents a single polygon from PPD file."""
    points: List[Tuple[int, int]]

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return (min_x, min_y, max_x, max_y)"""
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class PPDFile:
    """Represents a parsed PPD file."""
    version: int
    polygons: List[Polygon]

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Return overall bounds of all polygons."""
        if not self.polygons:
            return (0, 0, 0, 0)
        all_bounds = [p.bounds for p in self.polygons]
        return (
            min(b[0] for b in all_bounds),
            min(b[1] for b in all_bounds),
            max(b[2] for b in all_bounds),
            max(b[3] for b in all_bounds)
        )


def parse_ppd(filepath: str) -> PPDFile:
    """Parse a PPD file and return structured data."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 8:
        raise ValueError("File too small to be valid PPD")

    # Parse header
    version = struct.unpack('<H', data[0:2])[0]
    poly_count = struct.unpack('<H', data[2:4])[0]

    # Find polygon data start (skip padding zeros)
    offset = 8
    while offset < len(data) - 2:
        val = struct.unpack('<H', data[offset:offset+2])[0]
        if val != 0:
            break
        offset += 2

    # Parse polygons
    polygons = []
    for _ in range(poly_count):
        if offset + 2 > len(data):
            break

        point_count = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        # Sanity check
        if point_count > 10000:
            break

        points = []
        for _ in range(point_count):
            if offset + 4 > len(data):
                break
            x = struct.unpack('<H', data[offset:offset+2])[0]
            y = struct.unpack('<H', data[offset+2:offset+4])[0]
            points.append((x, y))
            offset += 4

        if points:
            polygons.append(Polygon(points=points))

    return PPDFile(version=version, polygons=polygons)


def ppd_to_json(ppd: PPDFile) -> dict:
    """Convert PPD data to JSON-serializable dict."""
    return {
        'version': ppd.version,
        'polygon_count': len(ppd.polygons),
        'bounds': ppd.bounds,
        'polygons': [
            {
                'point_count': len(poly.points),
                'bounds': poly.bounds,
                'points': poly.points
            }
            for poly in ppd.polygons
        ]
    }


def visualize_ppd(ppd: PPDFile, output_path: str, scale: int = 10, padding: int = 20):
    """Create a visualization of PPD polygons as PNG."""
    if not HAS_PIL:
        raise RuntimeError("PIL/Pillow required for visualization")

    bounds = ppd.bounds
    width = (bounds[2] - bounds[0] + 1) * scale + padding * 2
    height = (bounds[3] - bounds[1] + 1) * scale + padding * 2

    # Ensure minimum size
    width = max(width, 100)
    height = max(height, 100)

    img = Image.new('RGBA', (width, height), (32, 32, 32, 255))
    draw = ImageDraw.Draw(img)

    # Colors for different polygons
    colors = [
        (255, 100, 100, 180),  # Red
        (100, 255, 100, 180),  # Green
        (100, 100, 255, 180),  # Blue
        (255, 255, 100, 180),  # Yellow
        (255, 100, 255, 180),  # Magenta
        (100, 255, 255, 180),  # Cyan
        (255, 180, 100, 180),  # Orange
        (180, 100, 255, 180),  # Purple
    ]

    for i, poly in enumerate(ppd.polygons):
        color = colors[i % len(colors)]

        # Convert points to screen coordinates
        screen_points = [
            (
                (p[0] - bounds[0]) * scale + padding,
                (p[1] - bounds[1]) * scale + padding
            )
            for p in poly.points
        ]

        if len(screen_points) >= 3:
            # Draw filled polygon
            draw.polygon(screen_points, fill=color, outline=(255, 255, 255, 255))
        elif len(screen_points) == 2:
            # Draw line
            draw.line(screen_points, fill=(255, 255, 255, 255), width=2)

        # Draw points
        for px, py in screen_points:
            draw.ellipse([px-3, py-3, px+3, py+3], fill=(255, 255, 255, 255))

    img.save(output_path, 'PNG')


def convert_file(input_path: str, output_path: str, output_format: str = 'json',
                 scale: int = 10, verbose: bool = False) -> bool:
    """Convert a single PPD file."""
    try:
        ppd = parse_ppd(input_path)

        if verbose:
            print(f"  Version: {ppd.version}")
            print(f"  Polygons: {len(ppd.polygons)}")
            print(f"  Bounds: {ppd.bounds}")

        if output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(ppd_to_json(ppd), f, indent=2)
        elif output_format == 'png':
            visualize_ppd(ppd, output_path, scale=scale)

        return True
    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, output_format: str = 'json',
                      scale: int = 10, verbose: bool = False):
    """Convert all PPD files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = list(input_path.glob("*.ppd")) + list(input_path.glob("*.PPD"))

    success = 0
    fail = 0
    ext = '.json' if output_format == 'json' else '.png'

    for f in files:
        out_file = output_path / (f.stem + ext)

        if verbose:
            print(f"[{success + fail + 1}/{len(files)}] {f.name}")

        if convert_file(str(f), str(out_file), output_format, scale, verbose):
            success += 1
        else:
            fail += 1

    print(f"\nConversion complete: {success} succeeded, {fail} failed")


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver PPD (Polygon Path Data) Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file to JSON
  %(prog)s input.ppd -o output.json

  # Visualize as PNG
  %(prog)s input.ppd -o output.png --format png

  # Convert all files in directory
  %(prog)s ./Map/ -o ./ppd_output/

  # Show file info
  %(prog)s input.ppd --info
"""
    )

    parser.add_argument("input", help="Input PPD file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory")
    parser.add_argument("-f", "--format", choices=['json', 'png'], default='json',
                        help="Output format (default: json)")
    parser.add_argument("-s", "--scale", type=int, default=10,
                        help="Scale factor for PNG visualization (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--info", action="store_true", help="Show file info only")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    if args.info:
        if input_path.is_file():
            ppd = parse_ppd(str(input_path))
            print(f"File: {input_path.name}")
            print(f"  Version: {ppd.version}")
            print(f"  Polygon count: {len(ppd.polygons)}")
            print(f"  Bounds: {ppd.bounds}")
            for i, poly in enumerate(ppd.polygons):
                print(f"  Polygon {i}: {len(poly.points)} points, bounds={poly.bounds}")
        else:
            ppd_count = len(list(input_path.glob("*.ppd")) + list(input_path.glob("*.PPD")))
            print(f"Directory: {input_path}")
            print(f"  PPD files: {ppd_count}")
        return 0

    if args.format == 'png' and not HAS_PIL:
        print("Error: PIL/Pillow is required for PNG visualization")
        print("Install with: pip install Pillow")
        return 1

    if not args.output:
        print("Error: Output path required (-o)")
        return 1

    if input_path.is_file():
        success = convert_file(str(input_path), args.output, args.format, args.scale, args.verbose)
        if success:
            print(f"Converted: {args.output}")
        return 0 if success else 1
    else:
        convert_directory(str(input_path), args.output, args.format, args.scale, args.verbose)
        return 0


if __name__ == "__main__":
    exit(main())
