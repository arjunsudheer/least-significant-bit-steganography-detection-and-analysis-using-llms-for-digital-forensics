"""
lsb_extractor.py — Reimplementation of RobinDavid/LSB-Steganography.

Algorithm (from LSBSteg.py source)
─────────────────────────────────────────
  Encoding:
    1. Load image with cv2 → shape (H, W, 3), channels in BGR order
    2. Store 64-bit payload length: binary_value(len(data), 64) — MSB first
    3. For each payload byte b: store binary_value(b, 8) — MSB first
    4. Each bit is stored in bit-0 (the true LSB, mask=1) of consecutive channel
       values, traversed in row-major order: H → W → channel(B,G,R)
    5. If all bit-0 slots are exhausted the mask advances to bit-1, etc.
       (Only occurs for large payloads; for typical text payloads, bit-0 suffices.)

  Decoding (what this module implements):
    1. Load image → flatten in H×W×C (BGR) order
    2. Extract bit-0 of every value → 1-D bitstream
    3. Read bits[0:64] → 64-bit big-endian integer → payload byte count L
    4. Read bits[64 : 64 + L*8] → reshape to (L, 8) → each row is one byte MSB-first
    5. Return as bytes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def decode_lsb_robindavid(image_path: str | Path) -> bytes:
    """
    Decode a binary payload embedded by RobinDavid/LSB-Steganography.

    Fully vectorised with numpy — completes in < 50 ms for a 512×512 PNG.

    Returns the raw payload bytes, or b'' if the length header is invalid
    (e.g. the image was not stego-encoded with this library).
    """
    # Load as BGR (matching cv2.imread memory layout)
    img_rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    # PIL gives R,G,B per pixel; reverse last axis to get B,G,R
    img_bgr = img_rgb[:, :, ::-1]

    # Flatten: (H*W*3,) in B,G,R row-major order
    flat = img_bgr.flatten()

    # Extract bit-0 (the LSB, maskONE=1) of every channel value
    bits = (flat & np.uint8(1)).astype(np.uint8)

    # Read the 64-bit length header (MSB-first)
    if len(bits) < 64:
        return b""

    header = bits[:64].astype(np.uint64)
    powers_64 = np.uint64(1) << np.arange(63, -1, -1, dtype=np.uint64)
    length = int((header * powers_64).sum())

    # Sanity-check the declared length
    max_bytes = (len(bits) - 64) // 8
    if length == 0 or length > max_bytes:
        # Not a RobinDavid-encoded image, or carrier too small
        return b""

    # Extract payload bits and pack into bytes (MSB-first)
    payload_bits = bits[64 : 64 + length * 8].reshape(length, 8).astype(np.uint16)
    powers_8 = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
    payload = (payload_bits * powers_8).sum(axis=1).astype(np.uint8).tobytes()

    return payload


# Convenience wrappers
def extract_payload_bytes(image_path: str | Path) -> bytes:
    """Return the raw embedded payload bytes."""
    return decode_lsb_robindavid(image_path)


def extract_payload_text(
    image_path: str | Path,
    encoding: str = "utf-8",
) -> str:
    """
    Return the embedded payload decoded as text.

    Uses `errors='replace'` so non-UTF-8 payloads (binary files, encrypted
    data) produce a replacement-character string rather than raising an error.
    """
    raw = decode_lsb_robindavid(image_path)
    return raw.decode(encoding, errors="replace")


# Named wrappers used by pipeline.py and the LangGraph agent
def extract_lsb_robindavid_text(image_path: str) -> str:
    """Decode payload as UTF-8 text (replacement chars for non-text bytes)."""
    return extract_payload_text(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a payload embedded by RobinDavid/LSB-Steganography."
    )
    parser.add_argument("--image", required=True, help="Path to the stego image.")
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for the decoded payload (default: utf-8).",
    )
    parser.add_argument(
        "--raw", action="store_true", help="Write raw bytes instead of decoded text."
    )
    parser.add_argument(
        "--output", default=None, help="Write output to this file instead of stdout."
    )
    args = parser.parse_args()

    if args.raw:
        data: bytes | str = extract_payload_bytes(args.image)
    else:
        data = extract_payload_text(args.image, encoding=args.encoding)

    if args.output:
        if args.raw:
            Path(args.output).write_bytes(data)
        else:
            Path(args.output).write_text(data, encoding=args.encoding)
        print(f"Wrote payload to {args.output}")
    else:
        if args.raw:
            import sys

            sys.stdout.buffer.write(data)
        else:
            print(data)


if __name__ == "__main__":
    main()
