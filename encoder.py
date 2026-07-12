"""
QRGB Encoder
============

Splits an arbitrary file into sequential byte chunks, encodes each chunk as
base64 text inside a QR code, and superposes 3 QR codes (Red, Green, Blue
channels) into a single "QRGB" image. Images are numbered 0.png, 1.png, ...
in a chosen output directory, alongside a metadata.json describing how to
put the file back together.

Example: an 18-byte file with MAX_QR_CODE_SIZE = 3 bytes produces:
  - 18 bytes -> 6 chunks of 3 bytes each
  - 6 chunks -> grouped into pairs of 3 -> 2 QRGB images
  - each QRGB image = 3 QR codes (R, G, B), 3 bytes of original data each
    = 9 bytes of original file data per image

Run this file directly to launch the GUI.
"""

import os
import json
import base64
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image, ImageTk
import qrcode
from qrcode.constants import ERROR_CORRECT_H

DEFAULT_MAX_QR_CODE_SIZE = 950  # bytes per channel, per QR code
BOX_SIZE = 6
BORDER = 4


# --------------------------------------------------------------------------
# Core encoding logic
# --------------------------------------------------------------------------

def _required_version(data_bytes):
    """Base64-encode data_bytes and return (min QR version needed, b64 text)."""
    b64_text = base64.b64encode(data_bytes).decode("ascii")
    probe = qrcode.QRCode(error_correction=ERROR_CORRECT_H)
    probe.add_data(b64_text)
    probe.make(fit=True)
    return probe.version, b64_text


def _make_channel_qr(b64_text, version, fill_color):
    """Build a single-color QR image (any pixel != white counts as 'dark')."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECT_H,
        box_size=BOX_SIZE,
        border=BORDER,
    )
    qr.add_data(b64_text)
    qr.make(fit=False)
    return qr.make_image(fill_color=fill_color, back_color="white").convert("RGB")


def _combine_channels(img_r, img_g, img_b):
    """Superpose 3 same-size QR images into one RGB image.

    For each pixel, the output's R/G/B component is set to 255 if the
    corresponding source QR has a "dark" (non-white) module there, else 0.
    This is exactly reversible on the decode side.
    """
    if img_r.size != img_g.size or img_r.size != img_b.size:
        raise ValueError("Channel QR images must be the same pixel size")

    w, h = img_r.size
    r_px, g_px, b_px = img_r.load(), img_g.load(), img_b.load()

    out = Image.new("RGB", (w, h), (0, 0, 0))
    out_px = out.load()

    white = (255, 255, 255)
    for y in range(h):
        for x in range(w):
            red_active = r_px[x, y] != white
            green_active = g_px[x, y] != white
            blue_active = b_px[x, y] != white
            out_px[x, y] = (
                255 if red_active else 0,
                255 if green_active else 0,
                255 if blue_active else 0,
            )
    return out


def _encode_triplet(args):
    idx, triplet, output_dir = args
    r_bytes, g_bytes, b_bytes = triplet

    v_r, r_b64 = _required_version(r_bytes)
    v_g, g_b64 = _required_version(g_bytes)
    v_b, b_b64 = _required_version(b_bytes)

    version = max(v_r, v_g, v_b)

    # Generate all three QR codes simultaneously
    with ThreadPoolExecutor(max_workers=3) as qr_pool:
        futures = [
            qr_pool.submit(_make_channel_qr, r_b64, version, "red"),
            qr_pool.submit(_make_channel_qr, g_b64, version, "green"),
            qr_pool.submit(_make_channel_qr, b_b64, version, "blue"),
        ]
        img_r, img_g, img_b = [f.result() for f in futures]

    combined = _combine_channels(img_r, img_g, img_b)

    out_path = os.path.join(output_dir, f"{idx}.png")
    combined.save(out_path)

    return idx, out_path


def _chunk_file(file_bytes, chunk_size):
    if not file_bytes:
        return [b""]
    return [file_bytes[i:i + chunk_size] for i in range(0, len(file_bytes), chunk_size)]


def encode_file_to_qrgb(file_path, chunk_size, output_dir, progress_callback=None):
    """Encode a file into a directory of superposed QRGB PNGs + metadata.json.

    Returns (list_of_image_paths, metadata_dict).
    """
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    total_size = len(file_bytes)
    chunks = _chunk_file(file_bytes, chunk_size)

    # Group chunks into triplets of (red_bytes, green_bytes, blue_bytes)
    triplets = []
    for i in range(0, len(chunks), 3):
        triplet = chunks[i:i + 3]
        while len(triplet) < 3:
            triplet.append(b"")
        triplets.append(triplet)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_paths = [None] * len(triplets)


    workers = min(os.cpu_count() or 4, len(triplets))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_encode_triplet, (i, triplet, output_dir))
            for i, triplet in enumerate(triplets)
        ]

        completed = 0
        for future in as_completed(futures):
            idx, path = future.result()
            image_paths[idx] = path
            completed += 1
    
            if progress_callback:
                progress_callback(completed, len(triplets))

    metadata = {
        "original_filename": os.path.basename(file_path),
        "total_size": total_size,
        "chunk_size": chunk_size,
        "num_images": len(triplets),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return image_paths, metadata


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

class EncoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("QRGB Encoder")
        self.root.geometry("420x220")

        self.image_paths = []
        self.current_index = 0

        tk.Label(root, text="QRGB Encoder", font=(
            "Arial", 16, "bold")).pack(pady=10)
        tk.Button(
            root, text="Select File to Encode", command=self.select_file, width=25
        ).pack(pady=10)
        self.status_label = tk.Label(
            root, text="", wraplength=390, justify="left")
        self.status_label.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select a file to encode")
        if not file_path:
            return

        chunk_size = simpledialog.askinteger(
            "Chunk Size",
            "Enter MAX_QR_CODE_SIZE\n(bytes per channel, per QR code):",
            initialvalue=DEFAULT_MAX_QR_CODE_SIZE,
            minvalue=1,
        )
        if not chunk_size:
            return

        base_dir = os.path.dirname(os.path.abspath(file_path))
        stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(base_dir, f"{stem}_qrgb")

        self.status_label.config(text="Encoding, please wait...")
        self.root.update()

        def progress(done, total):
            self.status_label.config(
                text=f"Encoding QRGB code {done}/{total}...")
            self.root.update_idletasks()

        try:
            image_paths, metadata = encode_file_to_qrgb(
                file_path, chunk_size, output_dir, progress_callback=progress
            )
        except Exception as e:
            messagebox.showerror(
                "Encoding Error",
                f"{e}\n\nTip: if this is a capacity error, try a smaller "
                f"MAX_QR_CODE_SIZE.",
            )
            self.status_label.config(text="")
            return

        self.image_paths = image_paths
        self.current_index = 0

        self.status_label.config(
            text=f"Done! {metadata['num_images']} QRGB code(s) saved to:\n{output_dir}"
        )
        self.show_viewer()

    def show_viewer(self):
        viewer = tk.Toplevel(self.root)
        viewer.title("QRGB Codes")

        img_label = tk.Label(viewer)
        img_label.pack(padx=10, pady=10)

        counter_label = tk.Label(viewer, text="")
        counter_label.pack()

        nav_frame = tk.Frame(viewer)
        nav_frame.pack(pady=10)

        def render():
            path = self.image_paths[self.current_index]
            img = Image.open(path)
            img = img.resize((350, 350), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            img_label.config(image=img_tk)
            img_label.image = img_tk  # keep a reference
            counter_label.config(
                text=(
                    f"QR {self.current_index + 1} of {len(self.image_paths)}: "
                    f"{os.path.basename(path)}"
                )
            )
            prev_btn.config(
                state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            next_btn.config(
                state=tk.NORMAL
                if self.current_index < len(self.image_paths) - 1
                else tk.DISABLED
            )

        def go_next():
            if self.current_index < len(self.image_paths) - 1:
                self.current_index += 1
                render()

        def go_prev():
            if self.current_index > 0:
                self.current_index -= 1
                render()

        prev_btn = tk.Button(nav_frame, text="< Previous",
                             command=go_prev, width=12)
        prev_btn.grid(row=0, column=0, padx=5)

        next_btn = tk.Button(nav_frame, text="Next >",
                             command=go_next, width=12)
        next_btn.grid(row=0, column=1, padx=5)

        render()


if __name__ == "__main__":
    root = tk.Tk()
    app = EncoderApp(root)
    root.mainloop()
