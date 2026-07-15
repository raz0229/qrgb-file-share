"""
QRGB Decoder
============

Reads a directory of superposed QRGB PNGs (produced by encoder.py) and 
reconstructs the original file.

- 0.png contains the embedded metadata (JSON).
- 1.png onward contain chunk headers <chunk_number> + base64 data.
For each QRGB image, the Red/Green/Blue channels are evaluated rapidly 
using numpy, decoded to base64 text, concatenated to form the payload,
and strictly assembled using the <chunk_number> header.

Run this file directly to launch the GUI.
"""

import os
import re
import json
import base64

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode


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
    """Try to read a QR's text payload from a 2D numpy array.

    pyzbar is used first since it's more reliable. 
    OpenCV's detector acts as a fallback.
    """
    results = zbar_decode(channel_arr)
    if results:
        return results[0].data.decode("utf-8")

    # Fallback: OpenCV's detector (works natively on 2D uint8 arrays)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(channel_arr)
    if points is not None:
        return data
    return None


def decode_qrgb_directory(directory, progress_callback=None):
    """Decode a directory of QRGB images back into the original file bytes.

    Returns (file_bytes, metadata_dict).
    """
    # 1. Read Metadata from 0.png
    img_0_path = os.path.join(directory, "0.png")
    if not os.path.exists(img_0_path):
        raise FileNotFoundError("0.png (metadata) not found in the selected directory")

    # Each channel of 0.png is its OWN independently base64-encoded chunk
    # (encoder.py encodes r/g/b separately, with no header for metadata).
    # They must be base64-decoded individually and then have their raw
    # BYTES concatenated -- concatenating the base64 *text* across channels
    # before decoding is invalid, since each channel's base64 string has its
    # own padding ('=') that isn't just at the very end of the combined text.
    img_0_arr = np.array(Image.open(img_0_path).convert("RGB"))
    meta_bytes = bytearray()
    any_channel_read = False
    for ch_idx in range(3):
        ch_arr = _extract_channel_image(img_0_arr, ch_idx)
        txt = _read_qr_from_image(ch_arr)
        if not txt:
            continue
        any_channel_read = True

        # Clean this channel's payload: strip anything that isn't valid Base64
        cleaned_b64 = re.sub(r'[^A-Za-z0-9+/=]', '', txt)
        if not cleaned_b64:
            continue

        try:
            missing_padding = len(cleaned_b64) % 4
            if missing_padding:
                cleaned_b64 += '=' * (4 - missing_padding)
            meta_bytes.extend(base64.b64decode(cleaned_b64))
        except Exception as e:
            raise ValueError(
                f"Failed to decode Base64 metadata (channel {ch_idx}).\n"
                f"Error: {e}\n"
                f"Extracted Base64 Snippet: {cleaned_b64[:150]}..."
            )

    if not any_channel_read:
        raise ValueError("No text could be decoded from 0.png. The QR codes might be unreadable.")

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
    # THREE original chunks (R, G, B), each independently base64-encoded, so
    # they must be decoded separately rather than concatenated as text first.
    chunks = {}

    for i, fname in enumerate(png_files):
        img_path = os.path.join(directory, fname)
        img_arr = np.array(Image.open(img_path).convert("RGB"))

        channel_texts = []
        for ch_idx in range(3):
            ch_arr = _extract_channel_image(img_arr, ch_idx)
            channel_texts.append(_read_qr_from_image(ch_arr))
        r_txt, g_txt, b_txt = channel_texts

        if not r_txt:
            print(f"Warning: {fname} - Red channel (carries the chunk header) "
                  f"could not be read. Skipping image.")
            continue

        # Extract <image_number> header and Red channel's own base64 payload
        match = re.match(r'^<(\d+)>(.*)$', r_txt, re.DOTALL)
        if not match:
            print(f"Warning: {fname} is missing a valid chunk header. Skipping.")
            continue

        image_num = int(match.group(1))
        # This image's R/G/B channels hold original chunks base_idx,
        # base_idx+1, base_idx+2 respectively (see encoder.py triplet logic).
        base_idx = (image_num - 1) * 3
        r_b64 = match.group(2)

        for offset, txt in ((0, r_b64), (1, g_txt), (2, b_txt)):
            if txt is None:
                print(f"Warning: {fname} - channel for chunk "
                      f"{base_idx + offset} could not be read. Skipping chunk.")
                continue
            try:
                chunks[base_idx + offset] = base64.b64decode(txt) if txt else b""
            except Exception:
                print(f"Warning: Failed to decode base64 in chunk {base_idx + offset} ({fname}).")

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