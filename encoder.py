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
import io
import json
import base64
import shutil
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

from PIL import Image, ImageTk
from concurrent.futures import ProcessPoolExecutor, as_completed
import qrcode
from qrcode.constants import ERROR_CORRECT_L
from qrcode.util import QRData, MODE_8BIT_BYTE

DEFAULT_MAX_QR_CODE_SIZE = 2213  # bytes per channel, per QR code (ERROR_CORRECT_H=950, ERROR_CORRECT_L=2213)
BOX_SIZE = 6
BORDER = 4

# How much encoded image data we're willing to hold in memory before we
# flush it to disk. Keeps RAM usage bounded regardless of how many QRGB
# images the file produces.
DEFAULT_MEMORY_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MB


def _max_safe_chunk_size():
    """Largest number of raw bytes that can be base64-encoded and still fit
    into the biggest QR code this tool can produce (version 40, error-
    correction L, byte mode). base64 adds ~33% overhead, so this is well
    below the QR code's raw byte capacity.

    Computed from the qrcode library's own capacity tables so it stays
    correct if the library changes; falls back to a known-good constant
    (verified against qrcode's tables at the time of writing) if that
    introspection ever fails.
    """
    try:
        from qrcode import util as _qru, base as _qrbase
        version = 40
        bit_limit = sum(b.data_count * 8 for b in _qrbase.rs_blocks(version, ERROR_CORRECT_L))
        header_bits = 4 + _qru.length_in_bits(_qru.MODE_8BIT_BYTE, version)
        max_b64_chars = (bit_limit - header_bits) // 8
        best = 0
        for x in range(max_b64_chars, 0, -1):
            if ((x + 2) // 3) * 4 <= max_b64_chars:  # ceil(x/3)*4 <= capacity
                best = x
                break
        return best
    except Exception:
        return 2214  # known-good fallback


MAX_SAFE_CHUNK_SIZE = _max_safe_chunk_size()


# --------------------------------------------------------------------------
# Core encoding logic
# --------------------------------------------------------------------------

def _b64(data_bytes):
    """Base64-encode data_bytes."""
    return base64.b64encode(data_bytes).decode("ascii")


def _required_version_for_text(b64_text):
    """Return the minimum QR version needed to hold b64_text in byte mode.

    Mode is forced to MODE_8BIT_BYTE rather than left to the library's
    auto-detection. base64 output can occasionally be alphanumeric-mode-
    eligible (e.g. a run of zero bytes encodes as all 'A's), which needs
    fewer bits than byte mode for the same character count. Since we probe
    only the longest of the three channel strings and reuse its version for
    the other two, that reuse is only safe if bit-cost is a pure function of
    character count — which requires every channel to use the same mode.
    """
    probe = qrcode.QRCode(error_correction=ERROR_CORRECT_L)
    probe.add_data(QRData(b64_text, mode=MODE_8BIT_BYTE))
    probe.make(fit=True)
    return probe.version


def _make_channel_mask(b64_text, version):
    """Build a single-channel ('L') mask: white modules on black background."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECT_L,
        box_size=BOX_SIZE,
        border=BORDER,
    )
    qr.add_data(QRData(b64_text, mode=MODE_8BIT_BYTE))
    qr.make(fit=False)
    return qr.make_image(fill_color="white", back_color="black").convert("L")


def _chunk_file(file_bytes, chunk_size):
    if not file_bytes:
        return [b""]
    return [file_bytes[i:i + chunk_size] for i in range(0, len(file_bytes), chunk_size)]


def _encode_triplet_worker(args):
    idx, triplet = args
    r_bytes, g_bytes, b_bytes = triplet
    r_b64, g_b64, b_b64 = _b64(r_bytes), _b64(g_bytes), _b64(b_bytes)

    # Only probe once, using the longest of the three strings, instead of
    # probing (and discarding) a QR code per channel.
    longest = max((r_b64, g_b64, b_b64), key=len)
    version = _required_version_for_text(longest)

    mask_r = _make_channel_mask(r_b64, version)
    mask_g = _make_channel_mask(g_b64, version)
    mask_b = _make_channel_mask(b_b64, version)

    # Native PIL channel merge (runs in C) instead of numpy array math.
    combined = Image.merge("RGB", (mask_r, mask_g, mask_b))

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return idx, buf.getvalue()


def encode_file_to_qrgb(file_path, chunk_size, output_dir, progress_callback=None,
                         memory_threshold_bytes=DEFAULT_MEMORY_THRESHOLD_BYTES):
    if chunk_size > MAX_SAFE_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size ({chunk_size}) is too large: once base64-encoded it "
            f"would exceed the capacity of the largest QR code this tool can "
            f"generate. Max safe chunk_size is {MAX_SAFE_CHUNK_SIZE} bytes."
        )
    with open(file_path, "rb") as f: file_bytes = f.read()
    total_size = len(file_bytes)
    chunks = _chunk_file(file_bytes, chunk_size)
    triplets = []
    for i in range(0, len(chunks), 3):
        t = chunks[i:i + 3]
        while len(t) < 3: t.append(b"")
        triplets.append(t)
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_paths = [None] * len(triplets)

    # Images are kept in memory (as encoded PNG bytes) and only written to
    # disk once the buffered amount crosses memory_threshold_bytes, so disk
    # I/O happens in fewer, larger bursts instead of once per image, without
    # risking unbounded RAM growth.
    buffered = {}
    buffered_size = 0

    def flush_buffer():
        nonlocal buffered_size
        for i, data in buffered.items():
            path = os.path.join(output_dir, f"{i}.png")
            with open(path, "wb") as fh:
                fh.write(data)
            image_paths[i] = path
        buffered.clear()
        buffered_size = 0

    workers = min(os.cpu_count() or 1, len(triplets)) or 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_encode_triplet_worker, (i, t)) for i, t in enumerate(triplets)]
        done = 0
        for f in as_completed(futs):
            idx, data = f.result()
            buffered[idx] = data
            buffered_size += len(data)
            done += 1
            if progress_callback: progress_callback(done, len(triplets))
            if buffered_size >= memory_threshold_bytes:
                flush_buffer()

    flush_buffer()  # write out whatever's left in the buffer

    metadata = {"original_filename": os.path.basename(file_path), "total_size": total_size, "chunk_size": chunk_size, "num_images": len(triplets)}
    with open(os.path.join(output_dir, "metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
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

        tk.Label(root, text="QRGB Encoder", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Button(
            root, text="Select File to Encode", command=self.select_file, width=25
        ).pack(pady=10)
        self.status_label = tk.Label(root, text="", wraplength=390, justify="left")
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
            maxvalue=MAX_SAFE_CHUNK_SIZE,
        )
        if not chunk_size:
            return

        base_dir = os.path.dirname(os.path.abspath(file_path))
        stem = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = os.path.join(base_dir, f"{stem}_qrgb")

        self.status_label.config(text="Encoding, please wait...")
        self.root.update()

        def progress(done, total):
            self.status_label.config(text=f"Encoding QRGB code {done}/{total}...")
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
            prev_btn.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
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

        prev_btn = tk.Button(nav_frame, text="< Previous", command=go_prev, width=12)
        prev_btn.grid(row=0, column=0, padx=5)

        next_btn = tk.Button(nav_frame, text="Next >", command=go_next, width=12)
        next_btn.grid(row=0, column=1, padx=5)

        render()


if __name__ == "__main__":
    root = tk.Tk()
    app = EncoderApp(root)
    root.mainloop()