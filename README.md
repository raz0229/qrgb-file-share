# QRGB File Encoder / Decoder

Splits **any file** (binary or text) into sequential byte chunks, encodes
each chunk into a QR code, and superposes three QR codes (Red, Green, Blue)
into a single image — a "QRGB code". A full file becomes a numbered
sequence of these images (`0.png`, `1.png`, `2.png`, ...), which the
decoder can turn back into an exact copy of the original file.

## How the split works

Given `MAX_QR_CODE_SIZE` (bytes per channel, per QR code):

1. The file is split sequentially into chunks of `MAX_QR_CODE_SIZE` bytes.
2. Chunks are grouped into triplets → one triplet = one QRGB image
   (red = chunk *n*, green = chunk *n+1*, blue = chunk *n+2*).
3. Each chunk is base64-encoded and embedded as its own QR code; the three
   are superposed into one RGB PNG.

Example from the prompt: an 18-byte file with `MAX_QR_CODE_SIZE = 3`
produces 6 chunks of 3 bytes → grouped into 2 triplets → **2 QRGB images**,
each carrying 9 bytes of the original file (3 bytes per channel).

A `metadata.json` is written alongside the images recording the original
filename, total file size, chunk size, and image count — this is what lets
the decoder reconstruct the file byte-for-byte (including files whose size
isn't a clean multiple of the chunk size).

## Files

- `encoder.py` — GUI app. Pick a file, enter `MAX_QR_CODE_SIZE`, and it
  generates `<filename>_qrgb/` next to your file containing the numbered
  QRGB images + `metadata.json`. A viewer window opens automatically with
  **Next / Previous** buttons to page through the generated codes.
- `decoder.py` — GUI app. Pick the directory of QRGB images, and it
  reconstructs the original file and lets you save it wherever you like.

## Setup

```bash
pip install -r requirements.txt
```

`pyzbar` (used for QR decoding, more reliable than OpenCV's built-in
detector for these high-density codes) depends on the system library
**zbar**. If `pip install pyzbar` doesn't work out of the box:

- Ubuntu/Debian: `sudo apt-get install libzbar0`
- macOS: `brew install zbar`
- Windows: the bundled DLL in the `pyzbar` wheel is normally sufficient

## Usage

```bash
python encoder.py   # create QRGB codes from a file
python decoder.py   # reconstruct a file from a QRGB directory
```

## Notes / tuning

- `MAX_QR_CODE_SIZE` trades off image count vs. QR density. Smaller values
  → more, simpler QR images. Larger values → fewer, denser images (each QR
  uses error-correction level H and auto-selects the smallest QR version
  that fits the base64-encoded chunk, up to version 40). If you pick a
  chunk size so large that a channel can't fit in a version-40 QR code,
  the encoder will show an error — just lower `MAX_QR_CODE_SIZE` and retry.
- The three channels always share one QR *version* per image (the largest
  needed among R/G/B for that triplet), which is what keeps the three
  QR images the same pixel size so they can be superposed.
- Reconstruction is exact for any file type — it works on raw bytes, not
  text, so images, archives, executables, etc. all round-trip correctly.
