"""
QRGB Single Image Inspector
===========================

A standalone utility to read a single superimposed QRGB image and 
print the raw payload hidden inside its Red, Green, and Blue channels.

Usage:
    python inspect_qrgb.py path/to/image.png
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

def _extract_channel_image(img_arr, channel_index):
    """Rebuild a black/white 2D numpy array for one channel (0=R, 1=G, 2=B)."""
    channel = img_arr[:, :, channel_index]
    
    # Create a fast 2D matrix initialized to white
    out = np.full(channel.shape, 255, dtype=np.uint8)
    # Set active pixels to black
    out[channel > 127] = 0
    
    return out

def _read_qr_from_image(channel_arr):
    """Try to read a QR's text payload from a 2D numpy array."""
    # Try pyzbar first
    results = zbar_decode(channel_arr)
    if results:
        return results[0].data.decode("utf-8")

    # Fallback to OpenCV
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(channel_arr)
    if points is not None:
        return data
    return None

def inspect_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    print(f"Inspecting QRGB: {image_path}")
    print("=" * 50)

    try:
        # Load image and convert to RGB array
        img_arr = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"Failed to open image: {e}")
        sys.exit(1)

    channel_names = ["Red", "Green", "Blue"]
    
    for ch_idx, color in enumerate(channel_names):
        ch_arr = _extract_channel_image(img_arr, ch_idx)
        text = _read_qr_from_image(ch_arr)
        
        print(f"--- [ {color} Channel ] ---")
        if text:
            # Truncate output for the console if it's massively long, 
            # but show enough to be useful for debugging.
            display_text = text if len(text) < 500 else text[:500] + "\n... [TRUNCATED]"
            print(display_text)
        else:
            print("(No QR code detected or channel is empty)")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_qrgb.py <path_to_image.png>")
        sys.exit(1)
        
    target_image = sys.argv[1]
    inspect_image(target_image)