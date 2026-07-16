"""
QRGB Encoder
============

Splits an arbitrary file into sequential byte chunks, encodes each chunk as
RAW BYTES directly inside a QR code (byte mode natively supports any 8-bit
value, so no base64 step is needed), and superposes 3 QR codes (Red, Green,
Blue channels) into a single "QRGB" image. Images are numbered 0.png,
1.png, ... in a chosen output directory, alongside a metadata.json
describing how to put the file back together.

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
import struct
import shutil
import hashlib
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

from PIL import Image, ImageTk
from concurrent.futures import ProcessPoolExecutor, as_completed
import qrcode
from qrcode.constants import ERROR_CORRECT_L
from qrcode.util import QRData, MODE_8BIT_BYTE

BOX_SIZE = 6
BORDER = 4

# How much encoded image data we're willing to hold in memory before we
# flush it to disk. Keeps RAM usage bounded regardless of how many QRGB
# images the file produces.
DEFAULT_MEMORY_THRESHOLD_BYTES = 200 * 1024 * 1024  # 200 MB

# The Red channel of each data QRGB image is prefixed with a fixed-size
# binary header giving the image number, so the decoder can tell which
# original chunks that image's R/G/B channels correspond to without any
# text parsing. 4 bytes (up to ~4.29 billion images) is far more than
# enough and keeps the header overhead negligible and constant-size.
HEADER_STRUCT = struct.Struct(">I")  # big-endian, unsigned 4-byte int
HEADER_SIZE = HEADER_STRUCT.size

# QR scanners (zbar, and OpenCV's detector) don't treat byte-mode QR
# payloads as truly opaque binary: internally they check whether the bytes
# happen to already be valid UTF-8. If so, they hand them back unchanged;
# if not, they reinterpret every byte as a Latin-1 codepoint and re-encode
# it as UTF-8 before handing it back. Both behaviors are individually
# reversible, but the decoder can't tell *which one* happened just by
# looking at the result -- and getting it wrong silently corrupts data
# (e.g. this bites any chunk that happens to contain valid UTF-8 text).
# Prefixing every channel's payload with a single 0xFF byte -- a byte
# value that can never appear anywhere in valid UTF-8 -- guarantees the
# payload is *always* invalid UTF-8, so scanners always take the Latin-1
# reinterpretation path. That makes the transform deterministic and
# reversible on every chunk, for a negligible 1-byte-per-channel cost
# (versus base64's ~33% overhead).
SENTINEL = b"\xff"
SENTINEL_SIZE = len(SENTINEL)

# The `qrcode` library's Reed-Solomon error-correction step splits the
# payload into fixed-size blocks (~118-149 bytes each for version 40) and
# builds a GF(256) polynomial out of each block's raw bytes. If a block
# ever ends up being ALL 0x00 bytes, the library's polynomial code strips
# what it thinks are leading zero coefficients, finds the whole thing is
# zero, and later calls glog(0) -- which is mathematically undefined and
# raises `ValueError: glog(0)`.
#
# SENTINEL only guarantees the *first* byte of a channel's data is
# non-zero; everything after that is raw file bytes, which for real files
# (padding, sparse regions, zeroed buffers, aligned binary sections, etc.)
# very often contains long runs of 0x00 -- especially once a file is large
# enough to have several such regions. That's why small test files work
# fine but larger/real-world files intermittently crash with glog(0).
#
# Fix: XOR every payload byte with a fixed, deterministic pseudorandom
# keystream ("whitening") before it's handed to the QR encoder. This makes
# it astronomically unlikely (~256^-118) for any block to end up all-zero,
# regardless of what patterns exist in the source file, while remaining
# perfectly reversible (XOR-ing the whitened bytes with the same keystream
# a second time restores the originals). The keystream is derived from
# SHA-256 rather than the `random` module so it's guaranteed to produce the
# exact same bytes on every machine/Python version -- required since the
# decoder has to regenerate an identical keystream to undo this.
_WHITEN_SEED = b"QRGB-WHITEN-V1"


def _keystream(length):
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.sha256(_WHITEN_SEED + counter.to_bytes(4, "big")).digest()
        counter += 1
    return bytes(out[:length])


def _whiten(data):
    """XOR `data` with a deterministic keystream. Self-inverse: calling
    this again on the output restores the original bytes. IMPORTANT: the
    decoder must apply this exact same function to each channel's payload
    (after stripping the SENTINEL byte and, for the Red channel, the
    4-byte header) before writing bytes back out to the reconstructed
    file, or the output will be garbage."""
    if not data:
        return data
    return bytes(a ^ b for a, b in zip(data, _keystream(len(data))))


def _max_safe_chunk_size():
    """Largest number of raw bytes that can fit directly (no base64) into
    the biggest QR code this tool can produce (version 40, error-
    correction L, byte mode), after reserving room for the fixed-size
    binary header prepended to the Red channel.

    Computed from the qrcode library's own capacity tables so it stays
    correct if the library changes; falls back to a known-good constant
    (verified against qrcode's tables at the time of writing) if that
    introspection ever fails.
    """
    try:
        from qrcode import util as _qru, base as _qrbase
        version = 40
        bit_limit = sum(
            b.data_count * 8 for b in _qrbase.rs_blocks(version, ERROR_CORRECT_L))
        header_bits = 4 + _qru.length_in_bits(_qru.MODE_8BIT_BYTE, version)
        byte_capacity = (bit_limit - header_bits) // 8
        return byte_capacity - SENTINEL_SIZE - HEADER_SIZE
    except Exception:
        return 2953 - SENTINEL_SIZE - HEADER_SIZE  # known-good fallback (v40-L byte capacity)


MAX_SAFE_CHUNK_SIZE = _max_safe_chunk_size()

# Reserve a small margin below the true max so default runs comfortably.
DEFAULT_MAX_QR_CODE_SIZE = MAX_SAFE_CHUNK_SIZE


# --------------------------------------------------------------------------
# Core encoding logic
# --------------------------------------------------------------------------

def _make_channel_mask(data_bytes, version):
    """Build a single-channel ('L') mask: white modules on black background."""
    qr = qrcode.QRCode(
        version=version,
        error_correction=ERROR_CORRECT_L,
        box_size=BOX_SIZE,
        border=BORDER,
    )
    qr.add_data(QRData(data_bytes, mode=MODE_8BIT_BYTE))
    qr.make(fit=False)
    return qr.make_image(fill_color="white", back_color="black").convert("L")


def _chunk_file(file_bytes, chunk_size):
    if not file_bytes:
        return [b""]
    return [file_bytes[i:i + chunk_size] for i in range(0, len(file_bytes), chunk_size)]


def _encode_triplet_worker(args, isMetaData=False):
    idx, triplet = args
    r_bytes, g_bytes, b_bytes = triplet

    # Whiten each channel's payload first (see comment above) so a block of
    # raw file bytes can never come out all-zero and trip the QR library's
    # glog(0) bug. Then, every channel gets the SENTINEL byte prefixed, so
    # scanners always take the deterministic, reversible decode path. Only
    # the Red channel additionally gets the QRGB image number as a
    # fixed-size binary header (no text/base64 involved).
    r_payload = _whiten(r_bytes)
    g_payload = _whiten(g_bytes)
    b_payload = _whiten(b_bytes)

    r_data = SENTINEL + (HEADER_STRUCT.pack(idx + 1) + r_payload if not isMetaData else r_payload)
    g_data = SENTINEL + g_payload
    b_data = SENTINEL + b_payload
    
    # Only probe once, using the longest of the three byte-strings, instead
    # of probing (and discarding) a QR code per channel.
    longest = max((r_data, g_data, b_data), key=len)
    version = 40

    mask_r = _make_channel_mask(r_data, version)
    mask_g = _make_channel_mask(g_data, version)
    mask_b = _make_channel_mask(b_data, version)

    # Native PIL channel merge (runs in C) instead of numpy array math.
    combined = Image.merge("RGB", (mask_r, mask_g, mask_b))

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return idx, buf.getvalue()


def encode_file_to_qrgb(file_path, chunk_size, output_dir, progress_callback=None,
                        memory_threshold_bytes=DEFAULT_MEMORY_THRESHOLD_BYTES):
    if chunk_size > MAX_SAFE_CHUNK_SIZE:
        raise ValueError(
            f"chunk_size ({chunk_size}) is too large: it would exceed the "
            f"capacity of the largest QR code this tool can generate (plus "
            f"header overhead). Max safe chunk_size is {MAX_SAFE_CHUNK_SIZE} bytes."
        )
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    total_size = len(file_bytes)
    chunks = _chunk_file(file_bytes, chunk_size)
    triplets = []
    for i in range(0, len(chunks), 3):
        t = chunks[i:i + 3]
        while len(t) < 3:
            t.append(b"")
        triplets.append(t)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
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
            path = os.path.join(output_dir, f"{i+1}.png")
            with open(path, "wb") as fh:
                fh.write(data)
            image_paths[i] = path
        buffered.clear()
        buffered_size = 0

    workers = min(os.cpu_count() or 1, len(triplets)) or 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_encode_triplet_worker, (i, t))
                for i, t in enumerate(triplets)]
        done = 0
        for f in as_completed(futs):
            idx, data = f.result()
            buffered[idx] = data
            buffered_size += len(data)
            done += 1
            if progress_callback:
                progress_callback(done, len(triplets))
            if buffered_size >= memory_threshold_bytes:
                flush_buffer()

    flush_buffer()  # write out whatever's left in the buffer

    metadata = {
        "original_filename": os.path.basename(file_path),
        "total_size": total_size,
        "chunk_size": chunk_size,
        "num_images": len(triplets),
    }


    metadata_bytes = json.dumps(metadata, separators=(",", ":")).encode("utf-8")

    # Metadata should comfortably fit in a single QRGB.
    meta_chunks = _chunk_file(metadata_bytes, chunk_size)
    if len(meta_chunks) > 3:
        raise ValueError(
            "Metadata is too large to fit into a single QRGB image."
        )
    while len(meta_chunks) < 3:
        meta_chunks.append(b"")

    _, meta_png = _encode_triplet_worker((0, meta_chunks[:3]), isMetaData=True)

    with open(os.path.join(output_dir, "0.png"), "wb") as f:
        f.write(meta_png)
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