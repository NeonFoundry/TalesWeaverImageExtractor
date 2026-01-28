#!/usr/bin/env python3
"""
TalesWeaver HDT Sprite Converter v2 by NeonFoundry 2026

Based on reverse-engineered format specification.

HDT Format:
- Byte 0: Version
- Byte 1: BPPType (0=32BPP, 1=24BPP, 2=8BPP indexed)
- Bytes 2-5: Image count (int32)

Type 2 (8BPP Indexed):
- Palette at offset 6: 256 colors * 2 bytes (RGB565)
- Image headers at offset 0x206
- 8-bit indexed pixel data
"""

import struct
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def rgb565_to_rgb888(color: int) -> Tuple[int, int, int]:
    """Convert RGB565 to RGB888."""
    r5 = (color >> 11) & 0x1f
    g6 = (color >> 5) & 0x3f
    b5 = color & 0x1f

    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)

    return (r8, g8, b8)


@dataclass
class ImageInfo:
    """Image info from HDT header."""
    idx: int
    left: int
    top: int
    width: int
    height: int
    data_offset: int
    param_count: int


@dataclass
class HDTFile:
    """Parsed HDT file."""
    version: int
    bpp_type: int
    palette: List[Tuple[int, int, int]]  # RGB888 colors
    images: List[ImageInfo]
    raw_data: bytes


def parse_hdt(filepath: str) -> HDTFile:
    """Parse HDT file based on discovered format."""
    with open(filepath, 'rb') as f:
        data = f.read()

    if len(data) < 6:
        raise ValueError("File too small")

    version = data[0]
    bpp_type = data[1]
    img_count = struct.unpack('<I', data[2:6])[0]

    # Sanity check
    if img_count > 10000:
        # Might be old format with uint16 count
        img_count = struct.unpack('<H', data[2:4])[0]

    palette = []
    images = []

    if bpp_type == 2:
        # Type 2: 8BPP indexed
        # Palette at offset 6, 256 colors * 2 bytes (RGB565)
        palette_offset = 6
        for i in range(256):
            if palette_offset + 2 <= len(data):
                color = struct.unpack('<H', data[palette_offset:palette_offset+2])[0]
                palette.append(rgb565_to_rgb888(color))
                palette_offset += 2
            else:
                palette.append((0, 0, 0))

        # Image headers start at 0x206 (6 + 512)
        header_offset = 0x206

        for i in range(img_count):
            if header_offset + 0x1E > len(data):
                break

            # Parse header
            width = struct.unpack('<I', data[header_offset + 0x0E:header_offset + 0x12])[0]
            height = struct.unpack('<I', data[header_offset + 0x12:header_offset + 0x16])[0]
            param_count = struct.unpack('<I', data[header_offset + 0x16:header_offset + 0x1A])[0]
            data_offset = struct.unpack('<I', data[header_offset + 0x1A:header_offset + 0x1E])[0]

            # Sanity checks
            if width > 4096 or height > 4096 or param_count > 10000:
                break

            images.append(ImageInfo(
                idx=i,
                left=0,
                top=0,
                width=width,
                height=height,
                data_offset=data_offset,
                param_count=param_count
            ))

            # Next header offset
            param_size = param_count * 12
            header_offset += param_size + 0x1E

    elif bpp_type == 0:
        # Type 0: 32BPP - different structure
        # Image info: img_count * 40 bytes starting at offset 6
        info_offset = 6

        for i in range(img_count):
            if info_offset + 40 > len(data):
                break

            idx = struct.unpack('<I', data[info_offset:info_offset+4])[0]
            left = struct.unpack('<I', data[info_offset+4:info_offset+8])[0]
            top = struct.unpack('<I', data[info_offset+8:info_offset+12])[0]
            width = struct.unpack('<I', data[info_offset+12:info_offset+16])[0]
            height = struct.unpack('<I', data[info_offset+16:info_offset+20])[0]

            if width > 4096 or height > 4096:
                break

            images.append(ImageInfo(
                idx=idx,
                left=left,
                top=top,
                width=width,
                height=height,
                data_offset=0,
                param_count=0
            ))

            info_offset += 40

    return HDTFile(
        version=version,
        bpp_type=bpp_type,
        palette=palette,
        images=images,
        raw_data=data
    )


def extract_image_type2(hdt: HDTFile, img_idx: int, header_list: List[int],
                        img_data_offset: int) -> Optional[Image.Image]:
    """Extract a single image from Type 2 (8BPP indexed) HDT.

    The pixel data uses a sparse format with param lists that specify
    where to place runs of pixels (RLE-like compression).
    """
    if img_idx >= len(hdt.images):
        return None

    img_info = hdt.images[img_idx]
    width = img_info.width
    height = img_info.height

    if width == 0 or height == 0:
        return None

    data = hdt.raw_data
    header_offset = header_list[img_idx]

    # Read param count and data offset from header
    param_count = struct.unpack('<I', data[header_offset + 0x16:header_offset + 0x1A])[0]
    data_off = struct.unpack('<I', data[header_offset + 0x1A:header_offset + 0x1E])[0]

    # Create image (transparent by default)
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Param list starts after header
    param_offset = header_offset + 0x1E

    # Pixel data offset
    img_off = img_data_offset + data_off

    # Read pixels using param list (sparse/RLE format)
    for p in range(param_count):
        p_off = param_offset + p * 12
        if p_off + 12 > len(data):
            break

        left = struct.unpack('<I', data[p_off:p_off+4])[0]
        top = struct.unpack('<I', data[p_off+4:p_off+8])[0]
        max_count = struct.unpack('<I', data[p_off+8:p_off+12])[0]

        # Read max_count pixels and place them
        for y in range(max_count):
            if img_off >= len(data):
                break

            # Calculate destination position
            raw_off = (width * top) + y + left

            # Read palette index
            pal_idx = data[img_off]
            img_off += 1

            # Convert to x, y coordinates
            px = raw_off % width
            py = raw_off // width

            if 0 <= px < width and 0 <= py < height:
                if pal_idx < len(hdt.palette):
                    r, g, b = hdt.palette[pal_idx]
                    # Index 0 is typically transparent
                    a = 0 if pal_idx == 0 else 255
                    img.putpixel((px, py), (r, g, b, a))

    return img


def extract_image_type0(hdt: HDTFile, img_idx: int) -> Optional[Image.Image]:
    """Extract image from Type 0 (32BPP) HDT."""
    if img_idx >= len(hdt.images):
        return None

    img_info = hdt.images[img_idx]
    width = img_info.width
    height = img_info.height
    left = img_info.left
    top = img_info.top

    if width == 0 or height == 0:
        return None

    # Calculate block variables
    raw_info_size = len(hdt.images) * 40
    data = hdt.raw_data

    block_var1_offset = 6 + raw_info_size
    if block_var1_offset + 8 > len(data):
        return None

    block_var1 = struct.unpack('<I', data[block_var1_offset:block_var1_offset+4])[0]
    block_var2 = struct.unpack('<I', data[block_var1_offset+4:block_var1_offset+8])[0]

    img_data_offset = block_var1_offset + 8

    # Create image
    img = Image.new('RGBA', (width, height))

    for y in range(height):
        for x in range(width):
            col = ((top + y) * block_var1) + x
            row = left + col
            mem_offset = img_data_offset + row * 2

            if mem_offset + 2 <= len(data):
                sh = struct.unpack('<H', data[mem_offset:mem_offset+2])[0]
                b0 = data[mem_offset]
                b1 = data[mem_offset + 1]

                # Convert from RGB565-like format to RGBA
                r = (sh >> 4) & 0xF0
                g = b0 & 0xF0
                b = (b0 << 4) & 0xF0
                a = b1 & 0xF0

                img.putpixel((x, y), (r, g, b, a if a > 0 else 255))

    return img


def convert_hdt(input_path: str, output_path: str, verbose: bool = False) -> bool:
    """Convert HDT file to PNG(s)."""
    try:
        hdt = parse_hdt(input_path)

        if verbose:
            print(f"  Version: {hdt.version}")
            print(f"  BPP Type: {hdt.bpp_type}")
            print(f"  Images: {len(hdt.images)}")
            if hdt.palette:
                print(f"  Palette: {len(hdt.palette)} colors")

        if not hdt.images:
            if verbose:
                print("  No images found")
            return False

        data = hdt.raw_data

        # Build header list and find image data offset for Type 2
        header_list = []
        img_data_offset = 0

        if hdt.bpp_type == 2:
            # Headers start at 0x206 (after palette)
            header_off = 0x206
            for i in range(len(hdt.images)):
                header_list.append(header_off)
                param_count = struct.unpack('<I', data[header_off + 0x16:header_off + 0x1A])[0]
                param_size = param_count * 12
                header_off += param_size + 0x1E

            # Image data starts after all headers + 4 byte marker
            img_data_offset = header_off + 4

        # Extract images
        output_base = Path(output_path)

        for i, img_info in enumerate(hdt.images):
            if verbose:
                print(f"  Image {i}: {img_info.width}x{img_info.height}")

            if hdt.bpp_type == 2:
                img = extract_image_type2(hdt, i, header_list, img_data_offset)
            elif hdt.bpp_type == 0:
                img = extract_image_type0(hdt, i)
            else:
                img = None

            if img:
                if len(hdt.images) == 1:
                    out_file = str(output_base)
                else:
                    out_file = str(output_base.parent / f"{output_base.stem}_{i}.png")

                img.save(out_file, 'PNG')
                if verbose:
                    print(f"    Saved: {out_file}")

        return True

    except Exception as e:
        if verbose:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver HDT Sprite Converter v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.hdt -o output.png
  %(prog)s ./extracted/ -o ./png_output/
  %(prog)s input.hdt --info
"""
    )

    parser.add_argument("input", help="Input HDT file or directory")
    parser.add_argument("-o", "--output", help="Output PNG file or directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--info", action="store_true", help="Show file info only")

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    if args.info:
        if input_path.is_file():
            hdt = parse_hdt(str(input_path))
            print(f"File: {input_path.name}")
            print(f"  Version: {hdt.version}")
            print(f"  BPP Type: {hdt.bpp_type} ({'32BPP' if hdt.bpp_type == 0 else '24BPP' if hdt.bpp_type == 1 else '8BPP indexed'})")
            print(f"  Image count: {len(hdt.images)}")
            if hdt.palette:
                print(f"  Palette colors: {len(hdt.palette)}")
            for i, img in enumerate(hdt.images):
                print(f"  Image {i}: {img.width}x{img.height}")
        return 0

    if not HAS_PIL:
        print("Error: PIL/Pillow required")
        return 1

    if not args.output:
        print("Error: Output path required (-o)")
        return 1

    if input_path.is_file():
        success = convert_hdt(str(input_path), args.output, args.verbose)
        return 0 if success else 1
    else:
        # Directory mode
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob("*.hdt")) + list(input_path.glob("*.HDT"))
        success = 0

        for f in files:
            out_file = output_dir / (f.stem + ".png")
            print(f"[{success+1}/{len(files)}] {f.name}")
            if convert_hdt(str(f), str(out_file), args.verbose):
                success += 1

        print(f"\nDone: {success}/{len(files)}")
        return 0


if __name__ == "__main__":
    exit(main())
