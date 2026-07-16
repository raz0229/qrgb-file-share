"""
QRGB Decoder
============

Reads a directory of superposed QRGB PNGs (produced by encoder.py) and 
reconstructs the original file.

- 0.png contains the embedded metadata (JSON), stored as raw bytes with no
  encoding step.
- 1.png onward contain a 4-byte binary image-number header on the Red
  channel followed by raw chunk bytes; Green and Blue carry raw chunk bytes
  directly (see encoder.py for the exact layout).
For each QRGB image, the Red/Green/Blue channels are decoded independently
(each is its own QR payload) and assembled using the image-number header.

Run this file directly to launch the GUI.
"""

import os
import json
import struct
import hashlib

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

# Must match encoder.py's constants exactly.
HEADER_STRUCT = struct.Struct(">I")
HEADER_SIZE = HEADER_STRUCT.size
SENTINEL = b"\xff"
SENTINEL_SIZE = len(SENTINEL)

# encoder.py XORs every channel's payload bytes (metadata bytes, and each
# R/G/B chunk's bytes -- but NOT the SENTINEL byte or the Red channel's
# 4-byte header) with a deterministic keystream before encoding, to avoid
# an upstream qrcode-library bug where an all-zero run of raw file bytes
# crashes the encoder. This is self-inverse: applying it a second time on
# the decoded bytes restores the originals. Must match encoder.py exactly.
_WHITEN_SEED = b"QRGB-WHITEN-V1"


def _keystream(length):
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(_WHITEN_SEED + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:length])


def _whiten(data):
    if not data:
        return data
    return bytes(a ^ b for a, b in zip(data, _keystream(len(data))))


# --------------------------------------------------------------------------
# Core decoding logic
# --------------------------------------------------------------------------

def _extract_channel_image(img_arr, channel_index):
    """Rebuild a black/white 2D numpy array for one channel (0=R, 1=G, 2=B).
    
    Using numpy directly to create a 2D grayscale array drastically speeds up
    processing since zbar and cv2 don't need 3-channel RGB PIL images.
    Dark (module present) pixels -> 0 (black). Everything else -> 255 (white).
    """
    channel = img_arr[:, :, channel_index]
    
    # Create a fast 2D matrix initialized to white
    out = np.full(channel.shape, 255, dtype=np.uint8)
    # Set active pixels to black
    out[channel > 127] = 0
    
    return out


def _read_qr_from_image(channel_arr):
    """Try to read a QR's raw byte payload from a 2D numpy array.

    Returns the exact raw bytes that encoder.py wrote into the QR code
    (with the leading SENTINEL byte already stripped off), or None if
    nothing could be read.

    Both zbar (via pyzbar) and OpenCV's detector treat MODE_8BIT_BYTE QR
    payloads as Latin-1 text internally: they hand back the bytes unchanged
    if those bytes already happen to form valid UTF-8, or -- if not --
    reinterpret every byte as a Latin-1 codepoint and re-encode it as
    UTF-8. Those two behaviors are individually reversible but otherwise
    indistinguishable from the output alone, which is exactly why
    encoder.py prefixes every channel with a SENTINEL byte that can never
    appear in valid UTF-8: it forces scanners to always take the Latin-1
    reinterpretation path, so decoding as UTF-8 and re-encoding as Latin-1
    is always the correct reversal here. pyzbar/zbar does this reliably;
    OpenCV's detector is used only as a last-resort fallback for codes zbar
    misses and is somewhat less consistent.
    """
    def _strip_sentinel(data_bytes):
        if data_bytes[:SENTINEL_SIZE] != SENTINEL:
            return None
        return data_bytes[SENTINEL_SIZE:]

    results = zbar_decode(channel_arr)
    if results:
        # zbar hands back the raw UTF-8-encoded ctypes buffer directly.
        try:
            recovered = results[0].data.decode("utf-8").encode("latin-1")
        except (UnicodeDecodeError, UnicodeEncodeError):
            return None
        return _strip_sentinel(recovered)

    # Fallback: OpenCV's detector (works natively on 2D uint8 arrays)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(channel_arr)
    if points is not None:
        # cv2 hands back a str whose codepoints are already the Latin-1
        # mapped byte values, so a single encode('latin-1') recovers bytes.
        try:
            recovered = data.encode("latin-1")
        except UnicodeEncodeError:
            return None
        return _strip_sentinel(recovered)
    return None


def decode_qrgb_directory(directory, progress_callback=None):
    """Decode a directory of QRGB images back into the original file bytes.

    Returns (file_bytes, metadata_dict).
    """
    # 1. Read Metadata from 0.png
    img_0_path = os.path.join(directory, "0.png")
    if not os.path.exists(img_0_path):
        raise FileNotFoundError("0.png (metadata) not found in the selected directory")

    # Each channel of 0.png carries its own raw-byte slice of the metadata
    # JSON directly (encoder.py splits metadata across r/g/b with no header
    # and no encoding step) -- just read each channel's bytes and
    # concatenate them in R, G, B order.
    img_0_arr = np.array(Image.open(img_0_path).convert("RGB"))
    meta_bytes = bytearray()
    any_channel_read = False
    for ch_idx in range(3):
        ch_arr = _extract_channel_image(img_0_arr, ch_idx)
        data = _read_qr_from_image(ch_arr)
        if data is None:
            continue
        any_channel_read = True
        meta_bytes.extend(_whiten(data))

    if not any_channel_read:
        raise ValueError("No data could be decoded from 0.png. The QR codes might be unreadable.")

    try:
        metadata = json.loads(bytes(meta_bytes).decode('utf-8'))
    except Exception as e:
        raise ValueError(
            f"Failed to parse metadata JSON.\n"
            f"Error: {e}\n"
            f"Decoded bytes snippet: {bytes(meta_bytes[:150])!r}..."
        )

    total_size = metadata.get("total_size")

    # 2. Find and process all data chunks (ignoring 0.png)
    png_files = [f for f in os.listdir(directory) if f.endswith('.png') and f != '0.png']
    if not png_files:
        raise FileNotFoundError("No data PNG images found (expected 1.png, 2.png, etc.).")

    # Keyed by the ORIGINAL (pre-triplet) chunk index -- i.e. the position of
    # this chunk in the file, not the image number. Each QRGB image encodes
    # THREE original chunks (R, G, B) as raw bytes in three independent QR
    # payloads, so each channel is read and used directly (no decoding step).
    chunks = {}

    for i, fname in enumerate(png_files):
        img_path = os.path.join(directory, fname)
        img_arr = np.array(Image.open(img_path).convert("RGB"))

        channel_data = []
        for ch_idx in range(3):
            ch_arr = _extract_channel_image(img_arr, ch_idx)
            channel_data.append(_read_qr_from_image(ch_arr))
        r_data, g_data, b_data = channel_data

        if r_data is None or len(r_data) < HEADER_SIZE:
            print(f"Warning: {fname} - Red channel (carries the chunk header) "
                  f"could not be read. Skipping image.")
            continue

        # Extract the 4-byte binary image-number header (not whitened) and
        # the Red channel's own chunk bytes (everything after the header,
        # which IS whitened and must be un-whitened here).
        (image_num,) = HEADER_STRUCT.unpack(r_data[:HEADER_SIZE])
        r_chunk = _whiten(r_data[HEADER_SIZE:])
        g_data = _whiten(g_data) if g_data is not None else None
        b_data = _whiten(b_data) if b_data is not None else None

        # This image's R/G/B channels hold original chunks base_idx,
        # base_idx+1, base_idx+2 respectively (see encoder.py triplet logic).
        base_idx = (image_num - 1) * 3

        for offset, chunk_bytes in ((0, r_chunk), (1, g_data), (2, b_data)):
            if chunk_bytes is None:
                print(f"Warning: {fname} - channel for chunk "
                      f"{base_idx + offset} could not be read. Skipping chunk.")
                continue
            chunks[base_idx + offset] = chunk_bytes

        if progress_callback:
            progress_callback(i + 1, len(png_files))

    if not chunks:
        raise ValueError("No valid data chunks could be decoded from the images.")

    # 3. Check for missing chunks
    min_chunk = min(chunks.keys())
    max_chunk = max(chunks.keys())
    
    # Create a set of what chunks SHOULD exist between the min and max found
    expected_chunks = set(range(min_chunk, max_chunk + 1))
    found_chunks = set(chunks.keys())
    missing_chunks = expected_chunks - found_chunks

    if missing_chunks:
        missing_sorted = sorted(list(missing_chunks))
        raise ValueError(f"Missing chunk numbers detected: {missing_sorted}\nAssembly aborted.")

    # 4. Assemble bytes chronologically
    all_bytes = bytearray()
    for chunk_idx in sorted(chunks.keys()):
        all_bytes.extend(chunks[chunk_idx])

    if total_size and len(all_bytes) > total_size:
        file_bytes = bytes(all_bytes[:total_size])
    else:
        file_bytes = bytes(all_bytes)

    return file_bytes, metadata


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class DecoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QRGB Decoder")
        self.root.geometry("420x180")

        tk.Label(root, text="QRGB Decoder", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Button(
            root,
            text="Select QRGB Directory",
            command=self.select_directory,
            width=25,
        ).pack(pady=10)
        self.status_label = tk.Label(root, text="", wraplength=390, justify="left")
        self.status_label.pack(pady=10)

    def select_directory(self):
        directory = filedialog.askdirectory(
            title="Select the directory containing QRGB codes"
        )
        if not directory:
            return

        self.status_label.config(text="Decoding, please wait...")
        self.root.update()

        def progress(done, total):
            self.status_label.config(text=f"Decoding QRGB file {done}/{total}...")
            self.root.update_idletasks()

        try:
            file_bytes, metadata = decode_qrgb_directory(
                directory, progress_callback=progress
            )
        except Exception as e:
            messagebox.showerror("Decoding Error", str(e))
            self.status_label.config(text="")
            return

        default_name = metadata.get("original_filename", "decoded_file")
        save_path = filedialog.asksaveasfilename(
            title="Save decoded file as", initialfile=default_name
        )
        if not save_path:
            self.status_label.config(text="Decoding complete, but file was not saved.")
            return

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        self.status_label.config(
            text=f"File successfully reconstructed and saved to:\n{save_path}"
        )
        messagebox.showinfo("Success", f"File saved to:\n{save_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DecoderApp(root)
    root.mainloop()