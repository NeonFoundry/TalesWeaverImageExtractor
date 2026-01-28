#!/usr/bin/env python3
"""
TalesWeaver BMP2/TGA2 Texture Converter by NeonFoundry 2026

Converts BMP2 and TGA2 texture files to PNG format.

BMP2/TGA2 Format (per XentaxWiki documentation):
- Bytes 0-3: Width (uint32 LE)
- Bytes 4-7: Height (uint32 LE)
- Byte 8: Flag/padding (usually 0x00)
- Bytes 9+: Pixel data in ARGB4444 format

ARGB4444 Pixel Format:
- Bits 15-12: Alpha (4 bits)
- Bits 11-8: Red (4 bits)
- Bits 7-4: Green (4 bits)
- Bits 3-0: Blue (4 bits)

Images are stored bottom-to-top (like standard BMP).
"""

import struct
import os
import argparse
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL/Pillow not found. Install with: pip install Pillow")


def argb4444_to_rgba8888(color: int) -> Tuple[int, int, int, int]:
    """Convert ARGB4444 to RGBA8888.

    TalesWeaver BMP2/TGA2 format uses ARGB4444:
    - Bits 15-12: Alpha (4 bits)
    - Bits 11-8: Red (4 bits)
    - Bits 7-4: Green (4 bits)
    - Bits 3-0: Blue (4 bits)
    """
    a4 = (color >> 12) & 0x0f
    r4 = (color >> 8) & 0x0f
    g4 = (color >> 4) & 0x0f
    b4 = color & 0x0f

    # Expand 4-bit to 8-bit (replicate high bits into low bits)
    a8 = (a4 << 4) | a4
    r8 = (r4 << 4) | r4
    g8 = (g4 << 4) | g4
    b8 = (b4 << 4) | b4

    return (r8, g8, b8, a8)


def convert_bmp2(input_path: str, output_path: str, verbose: bool = False) -> bool:
    """Convert a BMP2/TGA2 file to PNG."""
    try:
        with open(input_path, 'rb') as f:
            data = f.read()

        # Parse header
        width = struct.unpack('<I', data[0:4])[0]
        height = struct.unpack('<I', data[4:8])[0]
        flag = data[8]

        # Validate dimensions
        if width == 0 or width > 4096 or height == 0 or height > 4096:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        expected_size = 9 + width * height * 2
        if len(data) < expected_size:
            raise ValueError(f"File too small: {len(data)} < {expected_size}")

        if verbose:
            print(f"  Dimensions: {width}x{height}")
            print(f"  Flag: 0x{flag:02x}")

        # Create RGBA image (with alpha channel)
        img = Image.new('RGBA', (width, height))

        # Parse pixel data (bottom-to-top)
        pixel_data = data[9:]
        for y in range(height):
            for x in range(width):
                idx = (y * width + x) * 2
                if idx + 2 <= len(pixel_data):
                    color = struct.unpack('<H', pixel_data[idx:idx+2])[0]
                    rgba = argb4444_to_rgba8888(color)
                    # Flip Y coordinate (BMP is bottom-to-top)
                    img.putpixel((x, height - 1 - y), rgba)

        img.save(output_path, 'PNG')
        return True

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
        return False


def convert_directory(input_dir: str, output_dir: str, verbose: bool = False):
    """Convert all BMP2/TGA2 files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .bmp2 and .tga2 files
    files = list(input_path.glob("*.bmp2")) + list(input_path.glob("*.tga2"))

    success = 0
    fail = 0

    for f in files:
        out_file = output_path / (f.stem + ".png")

        if verbose:
            print(f"[{success + fail + 1}/{len(files)}] {f.name}")

        if convert_bmp2(str(f), str(out_file), verbose=verbose):
            success += 1
        else:
            fail += 1

    print(f"\nConversion complete: {success} succeeded, {fail} failed")


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver BMP2/TGA2 Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  %(prog)s input.bmp2 -o output.png

  # Convert all files in directory
  %(prog)s ./BMP/ -o ./png_output/

  # Show file info
  %(prog)s input.bmp2 --info
"""
    )

    parser.add_argument("input", help="Input BMP2/TGA2 file or directory")
    parser.add_argument("-o", "--output", help="Output PNG file or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--info", action="store_true", help="Show file info only")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    if args.info:
        if input_path.is_file():
            with open(input_path, 'rb') as f:
                data = f.read(16)

            width = struct.unpack('<I', data[0:4])[0]
            height = struct.unpack('<I', data[4:8])[0]
            flag = data[8]

            print(f"File: {input_path.name}")
            print(f"  Width: {width}")
            print(f"  Height: {height}")
            print(f"  Flag: 0x{flag:02x}")
            print(f"  Format: ARGB4444")
        else:
            bmp2_count = len(list(input_path.glob("*.bmp2")))
            tga2_count = len(list(input_path.glob("*.tga2")))
            print(f"Directory: {input_path}")
            print(f"  BMP2 files: {bmp2_count}")
            print(f"  TGA2 files: {tga2_count}")
        return 0

    if not HAS_PIL:
        print("Error: PIL/Pillow is required for conversion")
        print("Install with: pip install Pillow")
        return 1

    if not args.output:
        print("Error: Output path required (-o)")
        return 1

    if input_path.is_file():
        success = convert_bmp2(str(input_path), args.output, verbose=args.verbose)
        if success:
            print(f"Converted: {args.output}")
        return 0 if success else 1
    else:
        convert_directory(str(input_path), args.output, verbose=args.verbose)
        return 0


if __name__ == "__main__":
    exit(main())
