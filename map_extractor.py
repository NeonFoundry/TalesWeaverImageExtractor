#!/usr/bin/env python3
"""
TalesWeaver Map Data Extractor by NeonFoundry 2026

Extracts and parses map data files from TalesWeaver Map folder:
- PPD: Polygon Path Data (collision/walkable areas)
- DMM: Map Markers (location names and coordinates)
- DWM: Warp/Waypoint Data

Usage:
  python map_extractor.py ./Map/ -o ./map_output/
"""

import struct
import os
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ============== PPD (Polygon Path Data) ==============

@dataclass
class Polygon:
    points: List[Tuple[int, int]]

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        if not self.points:
            return (0, 0, 0, 0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class PPDFile:
    version: int
    polygons: List[Polygon]

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
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
    """Parse PPD polygon file."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 8:
        raise ValueError("File too small")

    version = struct.unpack('<H', data[0:2])[0]
    poly_count = struct.unpack('<H', data[2:4])[0]

    # Skip padding
    offset = 8
    while offset < len(data) - 2:
        if struct.unpack('<H', data[offset:offset+2])[0] != 0:
            break
        offset += 2

    polygons = []
    for _ in range(poly_count):
        if offset + 2 > len(data):
            break

        point_count = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

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


# ============== DMM (Map Markers) ==============

@dataclass
class MapMarker:
    category: str
    name: str
    x: int
    y: int
    x2: int = 0
    y2: int = 0


@dataclass
class DMMFile:
    markers: List[MapMarker]


def parse_dmm(filepath: str) -> DMMFile:
    """Parse DMM map marker file."""
    with open(filepath, 'rb') as f:
        data = f.read()

    markers = []
    parts = data.split(b'\xff\xff')

    for part in parts:
        if len(part) < 20:
            continue

        name_match = re.search(rb'\[([^\]]+)\]\s*([^\x00]+)', part)
        if not name_match:
            continue

        category = name_match.group(1).decode('utf-8', errors='ignore')
        name = name_match.group(2).decode('utf-8', errors='ignore').strip()

        # Coordinates at bytes 3, 5, 7 (with 00 padding between)
        x1 = part[3] if len(part) > 3 else 0
        y1 = part[5] if len(part) > 5 else 0
        x2 = part[7] if len(part) > 7 else 0
        y2 = part[9] if len(part) > 9 else 0

        markers.append(MapMarker(
            category=category,
            name=name,
            x=x1,
            y=y1,
            x2=x2,
            y2=y2
        ))

    return DMMFile(markers=markers)


# ============== DWM (Waypoint/Warp Data) ==============

@dataclass
class Waypoint:
    id: int
    x: int
    y: int
    flags: int


@dataclass
class DWMFile:
    waypoints: List[Waypoint]


def parse_dwm(filepath: str) -> DWMFile:
    """Parse DWM waypoint file."""
    with open(filepath, 'rb') as f:
        data = f.read()

    waypoints = []

    # DWM structure appears to be repeating 14-byte entries
    # Format: count(2) + id(2) + unknown(2) + x(2) + y(2) + flags(2) + padding(2)

    if len(data) < 2:
        return DWMFile(waypoints=[])

    offset = 0
    count = struct.unpack('<H', data[0:2])[0]
    offset = 2

    # Each entry is ~14 bytes
    entry_size = 14
    while offset + entry_size <= len(data):
        wp_id = struct.unpack('<H', data[offset:offset+2])[0]
        # Skip some unknown bytes
        x = struct.unpack('<H', data[offset+6:offset+8])[0]
        y = struct.unpack('<H', data[offset+8:offset+10])[0]
        flags = struct.unpack('<H', data[offset+10:offset+12])[0]

        waypoints.append(Waypoint(id=wp_id, x=x, y=y, flags=flags))
        offset += entry_size

        if len(waypoints) >= count:
            break

    return DWMFile(waypoints=waypoints)


# ============== Export Functions ==============

def to_json(obj) -> dict:
    """Convert dataclass to JSON-serializable dict."""
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field in obj.__dataclass_fields__:
            value = getattr(obj, field)
            result[field] = to_json(value)
        return result
    elif isinstance(obj, list):
        return [to_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    else:
        return obj


def visualize_map(ppd: Optional[PPDFile], dmm: Optional[DMMFile],
                  output_path: str, scale: int = 5, padding: int = 50):
    """Create combined map visualization."""
    if not HAS_PIL:
        raise RuntimeError("PIL/Pillow required")

    # Calculate bounds
    min_x, min_y, max_x, max_y = 0, 0, 100, 100

    if ppd and ppd.polygons:
        bounds = ppd.bounds
        min_x = min(min_x, bounds[0])
        min_y = min(min_y, bounds[1])
        max_x = max(max_x, bounds[2])
        max_y = max(max_y, bounds[3])

    if dmm and dmm.markers:
        for m in dmm.markers:
            min_x = min(min_x, m.x)
            min_y = min(min_y, m.y)
            max_x = max(max_x, m.x)
            max_y = max(max_y, m.y)

    width = (max_x - min_x + 1) * scale + padding * 2
    height = (max_y - min_y + 1) * scale + padding * 2

    img = Image.new('RGBA', (width, height), (32, 32, 32, 255))
    draw = ImageDraw.Draw(img)

    # Draw polygons
    if ppd:
        colors = [
            (100, 100, 255, 100),
            (100, 255, 100, 100),
            (255, 100, 100, 100),
            (255, 255, 100, 100),
        ]
        for i, poly in enumerate(ppd.polygons):
            color = colors[i % len(colors)]
            screen_points = [
                ((p[0] - min_x) * scale + padding,
                 (p[1] - min_y) * scale + padding)
                for p in poly.points
            ]
            if len(screen_points) >= 3:
                draw.polygon(screen_points, fill=color, outline=(200, 200, 200, 255))

    # Draw markers
    if dmm:
        for m in dmm.markers:
            sx = (m.x - min_x) * scale + padding
            sy = (m.y - min_y) * scale + padding

            # Draw marker point
            draw.ellipse([sx-5, sy-5, sx+5, sy+5], fill=(255, 255, 0, 255))

            # Draw label
            label = f"[{m.category}] {m.name}"
            draw.text((sx + 8, sy - 5), label, fill=(255, 255, 255, 255))

    img.save(output_path, 'PNG')


# ============== Main ==============

def process_map(map_id: str, input_dir: Path, output_dir: Path,
                output_format: str, scale: int, verbose: bool):
    """Process all files for a single map ID."""
    ppd_path = input_dir / f"{map_id}.ppd"
    dmm_path = input_dir / f"{map_id}.dmm"
    dwm_path = input_dir / f"{map_id}.dwm"

    ppd = None
    dmm = None
    dwm = None

    if ppd_path.exists():
        try:
            ppd = parse_ppd(str(ppd_path))
            if verbose:
                print(f"  PPD: {len(ppd.polygons)} polygons")
        except Exception as e:
            if verbose:
                print(f"  PPD Error: {e}")

    if dmm_path.exists():
        try:
            dmm = parse_dmm(str(dmm_path))
            if verbose:
                print(f"  DMM: {len(dmm.markers)} markers")
        except Exception as e:
            if verbose:
                print(f"  DMM Error: {e}")

    if dwm_path.exists():
        try:
            dwm = parse_dwm(str(dwm_path))
            if verbose:
                print(f"  DWM: {len(dwm.waypoints)} waypoints")
        except Exception as e:
            if verbose:
                print(f"  DWM Error: {e}")

    if not (ppd or dmm or dwm):
        return False

    # Output
    if output_format == 'json':
        result = {
            'map_id': map_id,
            'polygons': to_json(ppd) if ppd else None,
            'markers': to_json(dmm) if dmm else None,
            'waypoints': to_json(dwm) if dwm else None,
        }
        out_path = output_dir / f"{map_id}.json"
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)

    elif output_format == 'png':
        out_path = output_dir / f"{map_id}.png"
        visualize_map(ppd, dmm, str(out_path), scale=scale)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver Map Data Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all maps to JSON
  %(prog)s ./Map/ -o ./map_output/

  # Visualize as PNG
  %(prog)s ./Map/ -o ./map_output/ --format png

  # Extract single map
  %(prog)s ./Map/ -o ./map_output/ --map 0002
"""
    )

    parser.add_argument("input", help="Map directory")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-f", "--format", choices=['json', 'png'], default='json')
    parser.add_argument("-m", "--map", help="Specific map ID to extract")
    parser.add_argument("-s", "--scale", type=int, default=5, help="PNG scale factor")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    if args.format == 'png' and not HAS_PIL:
        print("Error: PIL/Pillow required for PNG output")
        return 1

    # Find all map IDs
    if args.map:
        map_ids = [args.map]
    else:
        ppd_files = list(input_dir.glob("*.ppd"))
        dmm_files = list(input_dir.glob("*.dmm"))
        dwm_files = list(input_dir.glob("*.dwm"))
        all_files = ppd_files + dmm_files + dwm_files
        map_ids = sorted(set(f.stem for f in all_files))

    print(f"Processing {len(map_ids)} maps...")

    success = 0
    for map_id in map_ids:
        if args.verbose:
            print(f"\nMap {map_id}:")
        if process_map(map_id, input_dir, output_dir, args.format, args.scale, args.verbose):
            success += 1

    print(f"\nDone: {success}/{len(map_ids)} maps processed")
    return 0


if __name__ == "__main__":
    exit(main())
