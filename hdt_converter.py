#!/usr/bin/env python3
"""
TalesWeaver HDT Texture Converter by NeonFoundry 2026

WARNING! This is a work in progress. It is not currently functional as I'm still checking how
the format handles multiple frames, sprite sheets etc!'


Converts HDT texture files to PNG format.

HDT Format Types:
- 05 00: Multi-sprite container with embedded sprites
- 05 01: Sprite animation/sequence format
- 05 02: Direct RGB565 pixel data (most common)
- 06 XX: Similar formats with variations

All formats use RGB565 color encoding (16-bit: RRRRRGGGGGGBBBBB)
"""

import struct
import os
import argparse
import math
from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass


# Check for PIL/Pillow
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not found. Install with: pip install Pillow")


def xrgb1555_to_rgb888(color: int) -> Tuple[int, int, int]:
    """Convert XRGB1555 to RGB888.

    TalesWeaver uses XRGB1555 format:
    - Bit 15: Unused/Alpha (ignored)
    - Bits 14-10: Red (5 bits)
    - Bits 9-5: Green (5 bits)
    - Bits 4-0: Blue (5 bits)
    """
    r5 = (color >> 10) & 0x1f
    g5 = (color >> 5) & 0x1f
    b5 = color & 0x1f

    # Expand to 8-bit
    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g5 << 3) | (g5 >> 2)
    b8 = (b5 << 3) | (b5 >> 2)

    return (r8, g8, b8)


def bgr565_to_rgb888(color: int) -> Tuple[int, int, int]:
    """Convert BGR565 to RGB888 (alternative format)."""
    b5 = (color >> 11) & 0x1f
    g6 = (color >> 5) & 0x3f
    r5 = color & 0x1f

    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)

    return (r8, g8, b8)


def rgb565_to_rgb888(color: int) -> Tuple[int, int, int]:
    """Convert RGB565 to RGB888 (legacy, not used by TalesWeaver)."""
    r5 = (color >> 11) & 0x1f
    g6 = (color >> 5) & 0x3f
    b5 = color & 0x1f

    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)

    return (r8, g8, b8)


def find_best_dimensions(pixel_count: int, width_hint: int = 0) -> Tuple[int, int]:
    """Find the best width x height for a given pixel count.

    Prioritizes dimensions that are close to square.

    Args:
        pixel_count: Total number of pixels
        width_hint: Optional hint for width (e.g., from count field)
    """
    if pixel_count <= 0:
        return (1, 1)

    # If width_hint > 1 and divides evenly, use it
    if width_hint > 1 and pixel_count % width_hint == 0:
        height = pixel_count // width_hint
        if 4 <= height <= 4096:
            return (width_hint, height)

    # Find the dimension closest to square root
    sqrt_val = int(math.sqrt(pixel_count))

    # Search outward from sqrt to find exact divisor
    best_w, best_h = pixel_count, 1

    for offset in range(sqrt_val + 1):
        # Try sqrt - offset
        w = sqrt_val - offset
        if w >= 4 and pixel_count % w == 0:
            h = pixel_count // w
            if 4 <= h <= 4096:
                best_w, best_h = max(w, h), min(w, h)
                break

        # Try sqrt + offset
        w = sqrt_val + offset
        if w >= 4 and pixel_count % w == 0:
            h = pixel_count // w
            if 4 <= h <= 4096:
                best_w, best_h = max(w, h), min(w, h)
                break

    return (best_w, best_h)


@dataclass
class HdtFile:
    """Represents a parsed HDT file."""
    format_magic: int
    format_type: int
    count: int
    width: int
    height: int
    pixels: List[Tuple[int, int, int]]  # RGB tuples
    frames: int = 1


def parse_hdt_0502(data: bytes) -> HdtFile:
    """
    Parse HDT format 05 02 (direct RGB565 pixels).

    Structure:
    - Bytes 0-1: 0x05 0x02 (magic)
    - Bytes 2-3: count (uint16)
    - Bytes 4-5: unknown
    - Bytes 6-69: 32-color palette (64 bytes, RGB565) - may not be used
    - Bytes 70+: pixel data (RGB565)
    """
    magic = data[0]
    fmt_type = data[1]
    count = struct.unpack('<H', data[2:4])[0]

    # Pixel data starts at offset 70
    pixel_start = 70
    pixel_data = data[pixel_start:]

    # Ensure even length
    if len(pixel_data) % 2 != 0:
        pixel_data = pixel_data[:-1]

    pixel_count = len(pixel_data) // 2
    width, height = find_best_dimensions(pixel_count)

    # Parse RGB565 pixels
    pixels = []
    for i in range(0, len(pixel_data), 2):
        color = struct.unpack('<H', pixel_data[i:i+2])[0]
        pixels.append(bgr565_to_rgb888(color))

    # Pad if necessary
    while len(pixels) < width * height:
        pixels.append((0, 0, 0))

    return HdtFile(
        format_magic=magic,
        format_type=fmt_type,
        count=count,
        width=width,
        height=height,
        pixels=pixels[:width * height],
        frames=count if count > 0 else 1
    )


def parse_hdt_0501(data: bytes) -> HdtFile:
    """
    Parse HDT format 05 01 (sprite sequence).

    Structure:
    - Bytes 0-1: 0x05 0x01 (magic)
    - Bytes 2-3: count (number of entries)
    - Bytes 4-7: unknown
    - Bytes 8-9: offset_x (signed int16)
    - Bytes 10-11: offset_y (signed int16)
    - ... more header data
    - Bytes 20-23: width (uint32)
    - Bytes 24-27: height (uint32)
    - Bytes 28-31: frame count or similar
    - Followed by frame table and pixel data
    """
    magic = data[0]
    fmt_type = data[1]
    count = struct.unpack('<H', data[2:4])[0]

    # Try to read dimensions
    width = struct.unpack('<I', data[20:24])[0] if len(data) > 24 else 32
    height = struct.unpack('<I', data[24:28])[0] if len(data) > 28 else 32
    frames = struct.unpack('<I', data[28:32])[0] if len(data) > 32 else 1

    # Validate dimensions
    if width > 4096 or width == 0:
        width = 32
    if height > 4096 or height == 0:
        height = 32
    if frames > 1000:
        frames = 1

    # Find pixel data - skip header and frame table
    header_size = 32 + frames * 16  # Estimate

    # Look for actual pixel data by finding RGB565 patterns
    # Skip obvious zeros and small values
    pixel_start = header_size
    while pixel_start < len(data) - 2:
        val = struct.unpack('<H', data[pixel_start:pixel_start+2])[0]
        # If it looks like a valid RGB565 color (not all zeros, not too small)
        if val > 0x1000:
            break
        pixel_start += 2

    pixel_data = data[pixel_start:]

    # Ensure even length
    if len(pixel_data) % 2 != 0:
        pixel_data = pixel_data[:-1]

    pixel_count = len(pixel_data) // 2

    # Recalculate dimensions if necessary
    if width * height * frames > pixel_count:
        width, height = find_best_dimensions(pixel_count)
        frames = 1

    # Parse pixels
    pixels = []
    for i in range(0, len(pixel_data), 2):
        color = struct.unpack('<H', pixel_data[i:i+2])[0]
        pixels.append(bgr565_to_rgb888(color))

    # Pad if necessary
    target_size = width * height * frames
    while len(pixels) < target_size:
        pixels.append((0, 0, 0))

    return HdtFile(
        format_magic=magic,
        format_type=fmt_type,
        count=count,
        width=width,
        height=height,
        pixels=pixels[:target_size],
        frames=frames
    )


def parse_hdt_0500(data: bytes) -> HdtFile:
    """
    Parse HDT format 05 00 (multi-sprite container).

    Structure similar to 0501 but with different header layout.
    """
    magic = data[0]
    fmt_type = data[1]
    count = struct.unpack('<H', data[2:4])[0]

    # Dimensions at offset 18-21 as uint16
    width = struct.unpack('<H', data[18:20])[0] if len(data) > 20 else 32
    height = struct.unpack('<H', data[22:24])[0] if len(data) > 24 else 32

    if width == 0 or width > 4096:
        width = 32
    if height == 0 or height > 4096:
        height = 32

    # Sprite entries start at offset 32, each 32 bytes
    sprite_table_size = count * 32 if count < 100 else 0
    pixel_start = 32 + sprite_table_size

    # Find actual pixel data
    while pixel_start < len(data) - 2:
        val = struct.unpack('<H', data[pixel_start:pixel_start+2])[0]
        if val > 0x1000:
            break
        pixel_start += 2

    pixel_data = data[pixel_start:]

    if len(pixel_data) % 2 != 0:
        pixel_data = pixel_data[:-1]

    pixel_count = len(pixel_data) // 2

    # Recalculate if needed
    if pixel_count < width * height:
        width, height = find_best_dimensions(pixel_count)

    pixels = []
    for i in range(0, len(pixel_data), 2):
        color = struct.unpack('<H', pixel_data[i:i+2])[0]
        pixels.append(bgr565_to_rgb888(color))

    while len(pixels) < width * height:
        pixels.append((0, 0, 0))

    return HdtFile(
        format_magic=magic,
        format_type=fmt_type,
        count=count,
        width=width,
        height=height,
        pixels=pixels[:width * height],
        frames=count if count > 0 else 1
    )


def parse_hdt_generic(data: bytes) -> HdtFile:
    """Generic parser for unknown HDT formats."""
    magic = data[0]
    fmt_type = data[1]
    count = struct.unpack('<H', data[2:4])[0] if len(data) > 4 else 1

    # Skip first 6 bytes of header, treat rest as pixels
    pixel_data = data[6:]

    if len(pixel_data) % 2 != 0:
        pixel_data = pixel_data[:-1]

    pixel_count = len(pixel_data) // 2
    width, height = find_best_dimensions(pixel_count)

    pixels = []
    for i in range(0, len(pixel_data), 2):
        color = struct.unpack('<H', pixel_data[i:i+2])[0]
        pixels.append(bgr565_to_rgb888(color))

    while len(pixels) < width * height:
        pixels.append((0, 0, 0))

    return HdtFile(
        format_magic=magic,
        format_type=fmt_type,
        count=count,
        width=width,
        height=height,
        pixels=pixels[:width * height],
        frames=1
    )


def parse_hdt(filepath: str) -> HdtFile:
    """Parse an HDT file and return structured data."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 4:
        raise ValueError("File too small to be valid HDT")

    magic = data[0]
    fmt_type = data[1]

    if magic == 0x05:
        if fmt_type == 0x02:
            return parse_hdt_0502(data)
        elif fmt_type == 0x01:
            return parse_hdt_0501(data)
        elif fmt_type == 0x00:
            return parse_hdt_0500(data)
    elif magic == 0x06:
        # Format 06 variants - similar to 05 02
        return parse_hdt_0502(data)

    # Fallback to generic
    return parse_hdt_generic(data)


def hdt_to_png(hdt: HdtFile, output_path: str):
    """Convert parsed HDT to PNG image."""
    if not HAS_PIL:
        raise RuntimeError("PIL/Pillow required for PNG conversion")

    # Create image
    img = Image.new('RGB', (hdt.width, hdt.height))

    # Set pixels
    for y in range(hdt.height):
        for x in range(hdt.width):
            idx = y * hdt.width + x
            if idx < len(hdt.pixels):
                img.putpixel((x, y), hdt.pixels[idx])

    img.save(output_path, 'PNG')


def convert_file(input_path: str, output_path: str, verbose: bool = False) -> bool:
    """Convert a single HDT file to PNG."""
    try:
        hdt = parse_hdt(input_path)
        hdt_to_png(hdt, output_path)

        if verbose:
            print(f"Converted: {input_path}")
            print(f"  Format: 0x{hdt.format_magic:02x}{hdt.format_type:02x}")
            print(f"  Dimensions: {hdt.width}x{hdt.height}")
            print(f"  Frames/Count: {hdt.frames}")

        return True
    except Exception as e:
        if verbose:
            print(f"Error converting {input_path}: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, verbose: bool = False):
    """Convert all HDT files in a directory to PNG."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hdt_files = list(input_path.glob("*.hdt"))
    success_count = 0
    fail_count = 0

    for hdt_file in hdt_files:
        out_file = output_path / (hdt_file.stem + ".png")

        if verbose:
            print(f"[{success_count + fail_count + 1}/{len(hdt_files)}] {hdt_file.name}...")

        if convert_file(str(hdt_file), str(out_file), verbose=False):
            success_count += 1
        else:
            fail_count += 1
            if verbose:
                print(f"  FAILED")

    print(f"\nConversion complete: {success_count} succeeded, {fail_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver HDT Texture Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  %(prog)s input.hdt -o output.png

  # Convert all files in directory
  %(prog)s ./extracted/ -o ./png_output/

  # Show file info without converting
  %(prog)s input.hdt --info
"""
    )

    parser.add_argument("input", help="Input HDT file or directory")
    parser.add_argument("-o", "--output", help="Output PNG file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--info", action="store_true", help="Show file info only")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    if args.info:
        # Just show info
        if input_path.is_file():
            hdt = parse_hdt(str(input_path))
            print(f"File: {input_path.name}")
            print(f"  Format: 0x{hdt.format_magic:02x}{hdt.format_type:02x}")
            print(f"  Count: {hdt.count}")
            print(f"  Dimensions: {hdt.width}x{hdt.height}")
            print(f"  Total pixels: {len(hdt.pixels)}")
        else:
            # Show stats for directory
            from collections import Counter
            formats = Counter()
            for f in input_path.glob("*.hdt"):
                with open(f, 'rb') as fp:
                    header = fp.read(2)
                formats[f"0x{header[0]:02x}{header[1]:02x}"] += 1

            print(f"Directory: {input_path}")
            print(f"Format distribution:")
            for fmt, count in formats.most_common():
                print(f"  {fmt}: {count} files")
        return 0

    if not HAS_PIL:
        print("Error: PIL/Pillow is required for conversion")
        print("Install with: pip install Pillow")
        return 1

    if not args.output:
        print("Error: Output path required (-o)")
        return 1

    if input_path.is_file():
        success = convert_file(str(input_path), args.output, verbose=args.verbose)
        return 0 if success else 1
    else:
        convert_directory(str(input_path), args.output, verbose=args.verbose)
        return 0


if __name__ == "__main__":
    exit(main())
