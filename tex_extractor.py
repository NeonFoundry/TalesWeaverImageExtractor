#!/usr/bin/env python3
"""
TalesWeaver TEX Archive Extractor by NeonFoundry 2026

TEX Archive Format:
===================

Index File (TEX.DAT):
---------------------
Header (24 bytes):
  - [12 bytes] Magic: "D2PACKFILE2\x01"
  - [4 bytes]  Unknown (big endian value?)
  - [4 bytes]  Possible flags or version
  - [4 bytes]  Number of entries (3305 in this archive, last is terminator)

Entry Structure (28 bytes each, starting at offset 0x18):
  - [4 bytes]  Constant marker (usually 0x08)
  - [4 bytes]  XOR'd file ID (XOR each byte with 0x85 to decode)
  - [4 bytes]  XOR'd extension marker (XOR with 0x85 -> "\x1etdh")
  - [4 bytes]  DAT file number (0-19 for TEX0.DAT through TEX19.DAT)
  - [4 bytes]  Offset within the DAT file
  - [4 bytes]  Uncompressed size
  - [4 bytes]  Compressed size

Last entry is a terminator with file_idx=9 and different XOR pattern.

Data Files (TEX0.DAT - TEX19.DAT):
----------------------------------
Raw zlib-compressed data at specified offsets. Each entry decompresses
to a custom texture format (HDT format).

Decompressed File Format:
-------------------------
Files start with a 2-byte signature:
  - 0x0500: Format type 0
  - 0x0501: Format type 1
  - 0x0502: Format type 2 (most common)
  - 0x0600: Format type 0 (version 6)
  - 0x0601: Format type 1 (version 6)
  - 0x0602: Format type 2 (version 6)

The actual image data format appears to be proprietary HDT format
used by TalesWeaver.
"""

import struct
import zlib
import os
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, BinaryIO


@dataclass
class TexEntry:
    """Represents a single file entry in the TEX archive."""
    index: int
    file_id: int
    dat_file_num: int
    offset: int
    uncompressed_size: int
    compressed_size: int

    @property
    def filename(self) -> str:
        """Generate filename from file ID."""
        # Decode the file ID into folder/file structure
        # ID format appears to be: folder_id * 0x10000 + file_id
        folder = (self.file_id >> 8) & 0xFFFF
        file_num = self.file_id & 0xFF
        high = (self.file_id >> 24) & 0xFF

        if high > 0:
            return f"{high:02x}_{folder:04x}_{file_num:02x}.hdt"
        elif folder > 0:
            return f"{folder:04x}_{file_num:02x}.hdt"
        else:
            return f"{file_num:02x}.hdt"


class TexArchive:
    """Parser for TalesWeaver TEX archives."""

    MAGIC = b"D2PACKFILE2\x01"
    ENTRY_SIZE = 28
    HEADER_SIZE = 24
    XOR_KEY = 0x85

    def __init__(self, tex_dir: str):
        """
        Initialize the archive parser.

        Args:
            tex_dir: Path to directory containing TEX.DAT and TEX*.DAT files
        """
        self.tex_dir = Path(tex_dir)
        self.index_path = self.tex_dir / "TEX.DAT"
        self.entries: List[TexEntry] = []
        self._dat_handles: dict[int, BinaryIO] = {}

    def _get_dat_path(self, num: int) -> Path:
        """Get path to a specific DAT file."""
        return self.tex_dir / f"TEX{num}.DAT"

    def _get_dat_handle(self, num: int) -> BinaryIO:
        """Get or open a file handle for a DAT file."""
        if num not in self._dat_handles:
            path = self._get_dat_path(num)
            self._dat_handles[num] = open(path, 'rb')
        return self._dat_handles[num]

    def close(self):
        """Close all open file handles."""
        for handle in self._dat_handles.values():
            handle.close()
        self._dat_handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def parse_index(self) -> int:
        """
        Parse the TEX.DAT index file.

        Returns:
            Number of entries parsed
        """
        with open(self.index_path, 'rb') as f:
            # Read and verify magic
            magic = f.read(12)
            if magic != self.MAGIC:
                raise ValueError(f"Invalid magic: {magic!r}, expected {self.MAGIC!r}")

            # Skip rest of header
            f.seek(self.HEADER_SIZE)

            # Read all entries
            entry_idx = 0
            while True:
                entry_data = f.read(self.ENTRY_SIZE)
                if len(entry_data) < self.ENTRY_SIZE:
                    break

                # Parse entry fields
                marker = struct.unpack('<I', entry_data[0:4])[0]
                xor_id = entry_data[4:8]
                xor_ext = entry_data[8:12]
                dat_file_num = struct.unpack('<I', entry_data[12:16])[0]
                offset = struct.unpack('<I', entry_data[16:20])[0]
                uncomp_size = struct.unpack('<I', entry_data[20:24])[0]
                comp_size = struct.unpack('<I', entry_data[24:28])[0]

                # Check for terminator entry (marker=9, different XOR pattern)
                if marker == 9:
                    break

                # Decode file ID by XORing with key
                file_id = int.from_bytes(
                    bytes([b ^ self.XOR_KEY for b in xor_id]),
                    'little'
                )

                entry = TexEntry(
                    index=entry_idx,
                    file_id=file_id,
                    dat_file_num=dat_file_num,
                    offset=offset,
                    uncompressed_size=uncomp_size,
                    compressed_size=comp_size
                )
                self.entries.append(entry)
                entry_idx += 1

        return len(self.entries)

    def extract_entry(self, entry: TexEntry) -> bytes:
        """
        Extract and decompress a single entry.

        Args:
            entry: The entry to extract

        Returns:
            Decompressed data
        """
        handle = self._get_dat_handle(entry.dat_file_num)
        handle.seek(entry.offset)
        compressed = handle.read(entry.compressed_size)

        return zlib.decompress(compressed)

    def extract_all(self, output_dir: str, verbose: bool = False):
        """
        Extract all files to the output directory.

        Args:
            output_dir: Directory to extract files to
            verbose: Print progress information
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(self.entries):
            filename = entry.filename
            out_file = output_path / filename

            if verbose:
                print(f"[{i+1}/{len(self.entries)}] Extracting {filename}...")

            try:
                data = self.extract_entry(entry)

                # Verify size
                if len(data) != entry.uncompressed_size:
                    print(f"  WARNING: Size mismatch for {filename}: "
                          f"got {len(data)}, expected {entry.uncompressed_size}")

                with open(out_file, 'wb') as f:
                    f.write(data)

            except Exception as e:
                print(f"  ERROR extracting {filename}: {e}")

    def list_files(self):
        """Print a list of all files in the archive."""
        print(f"{'Index':>6} {'File ID':>12} {'DAT#':>5} {'Offset':>12} "
              f"{'Comp':>10} {'Uncomp':>10} {'Filename'}")
        print("-" * 80)

        for entry in self.entries:
            print(f"{entry.index:>6} {entry.file_id:>12} {entry.dat_file_num:>5} "
                  f"{entry.offset:>12} {entry.compressed_size:>10} "
                  f"{entry.uncompressed_size:>10} {entry.filename}")


def main():
    parser = argparse.ArgumentParser(
        description="TalesWeaver TEX Archive Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all files in archive
  %(prog)s /path/to/Tex --list

  # Extract all files
  %(prog)s /path/to/Tex --output ./extracted

  # Extract specific file by index
  %(prog)s /path/to/Tex --index 0 --output ./extracted
"""
    )

    parser.add_argument("tex_dir", help="Path to Tex directory containing TEX.DAT")
    parser.add_argument("-o", "--output", help="Output directory for extracted files")
    parser.add_argument("-l", "--list", action="store_true", help="List files in archive")
    parser.add_argument("-i", "--index", type=int, help="Extract specific file by index")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    with TexArchive(args.tex_dir) as archive:
        num_entries = archive.parse_index()
        print(f"Parsed {num_entries} entries from TEX.DAT")

        if args.list:
            archive.list_files()
        elif args.output:
            if args.index is not None:
                # Extract single file
                if args.index < 0 or args.index >= len(archive.entries):
                    print(f"Error: Index {args.index} out of range (0-{len(archive.entries)-1})")
                    return 1

                entry = archive.entries[args.index]
                output_path = Path(args.output)
                output_path.mkdir(parents=True, exist_ok=True)

                data = archive.extract_entry(entry)
                out_file = output_path / entry.filename

                with open(out_file, 'wb') as f:
                    f.write(data)
                print(f"Extracted: {out_file}")
            else:
                # Extract all
                archive.extract_all(args.output, verbose=args.verbose)
                print(f"Extraction complete: {args.output}")
        else:
            parser.print_help()

    return 0


if __name__ == "__main__":
    exit(main())
