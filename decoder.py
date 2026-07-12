"""
QRGB Decoder
============

Reads a directory of superposed QRGB PNGs (produced by encoder.py, numbered
0.png, 1.png, ... plus a metadata.json) and reconstructs the original file.

For each QRGB image, the Red/Green/Blue channel is split back out into its
own black-on-white QR image (a pixel is "dark" for a channel if that
channel's component is 255), each is decoded to base64 text, base64-decoded
back to bytes, and the bytes from all images are concatenated in order
(Red, Green, Blue per image; images in numeric order) and truncated to the
original file size recorded in metadata.json.

Run this file directly to launch the GUI.
"""

import os
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

def _extract_channel_image(combined_img, channel_index):
    """Rebuild a black/white QR image for one channel (0=R, 1=G, 2=B).

    Dark (module present) pixels -> black. Everything else -> white.
    """
    arr = np.array(combined_img.convert("RGB"))
    channel = arr[:, :, channel_index]
    active = channel > 127

    out = np.full(arr.shape, 255, dtype=np.uint8)
    out[active] = [0, 0, 0]
    return Image.fromarray(out, "RGB")


def _read_qr_from_image(channel_img):
    """Try to read a QR's text payload from a black/white PIL image.

    pyzbar (libzbar) is used first since it's noticeably more reliable than
    OpenCV's built-in QRCodeDetector for these synthetically-generated,
    high-density codes. OpenCV is kept as a fallback in case zbar isn't
    available in some environment.
    """
    results = zbar_decode(channel_img)
    if results:
        return results[0].data.decode("utf-8")

    # Fallback: OpenCV's detector.
    cv_img = cv2.cvtColor(np.array(channel_img), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(cv_img)
    # NOTE: `data` can legitimately be an empty string (e.g. a padding
    # channel that carries no real file bytes), so we must not treat an
    # empty string as failure. `points is not None` is what indicates the
    # detector actually located and decoded a QR code.
    if points is not None:
        return data
    return None


def _decode_channel(combined_img, channel_index):
    channel_img = _extract_channel_image(combined_img, channel_index)
    return _read_qr_from_image(channel_img)


def decode_qrgb_directory(directory, progress_callback=None):
    """Decode a directory of QRGB images back into the original file bytes.

    Returns (file_bytes, metadata_dict).
    """
    metadata_path = os.path.join(directory, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("metadata.json not found in the selected directory")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    num_images = metadata["num_images"]
    total_size = metadata["total_size"]

    channel_names = ("red", "green", "blue")
    all_bytes = bytearray()

    for idx in range(num_images):
        img_path = os.path.join(directory, f"{idx}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing QRGB image: {img_path}")

        combined_img = Image.open(img_path)

        for channel_index, channel_name in enumerate(channel_names):
            b64_text = _decode_channel(combined_img, channel_index)
            if b64_text is None:
                raise ValueError(
                    f"Failed to decode {channel_name} channel of image {idx}.png"
                )
            chunk_bytes = base64.b64decode(b64_text) if b64_text else b""
            all_bytes.extend(chunk_bytes)

        if progress_callback:
            progress_callback(idx + 1, num_images)

    file_bytes = bytes(all_bytes[:total_size])
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
            self.status_label.config(text=f"Decoding QRGB code {done}/{total}...")
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
