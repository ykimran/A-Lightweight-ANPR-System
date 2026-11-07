# A Lightweight ANPR System

Automatic Number Plate Recognition using classical image processing for plate localization and **EasyOCR** for English text recognition.

Developed for **EEE-3604 · Digital Signal Processing I Sessional**, Department of EEE, International Islamic University Chittagong (IIUC), Autumn 2025.


## Overview

A vehicle image passes through grayscale conversion, bilateral filtering, Canny edge detection, and contour analysis. The first four-sided polygon among the ten largest contours becomes the plate candidate. EasyOCR reads the cropped region. If no candidate is found, the system attempts OCR on the full filtered image.

“Lightweight” describes the classical localization stage. EasyOCR uses deep learning and depends on PyTorch; this is not a neural-network-free system. No trained custom plate detector or training dataset is included.

## Quick start

Use Python 3.10 or newer. Python 3.11 is a suggested starting point.

```bash
git clone https://github.com/ykimran/A-Lightweight-ANPR-System.git
cd A-Lightweight-ANPR-System
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install and run:

```bash
python -m pip install -r requirements.txt
python scripts/create_demo_image.py
python -m anpr outputs/synthetic-demo.png --output outputs/demo-run
```

The first OCR run requires internet access to download EasyOCR model weights. CPU is the default. Add `--gpu` only with a compatible PyTorch/GPU installation. Dependencies are version ranges, not a frozen environment.

Each run saves `result.json`, an annotated image, grayscale, filtered, and edge images. When localization succeeds, it also saves the masked image and OCR crop. Use a different output directory for each run to avoid mixing stage images from earlier runs.

The JSON includes recognized text, per-detection OCR confidence and image-coordinate boxes, the candidate contour, and a `fallback` flag. OCR confidence is not measured plate-recognition accuracy. Full-image fallback can include unrelated scene text.

## Run in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/ykimran/A-Lightweight-ANPR-System/blob/main/notebooks/anpr_demo.ipynb). Run the installation cell, upload a vehicle image, and execute the remaining cells. The notebook uses the same pipeline as the command-line app and displays intermediate stages.

## Repository guide

| Path | Contents |
| --- | --- |
| `src/anpr/` | Reusable pipeline and command-line entry point |
| `notebooks/anpr_demo.ipynb` | Colab upload and visualization workflow |
| `docs/methodology.md` | Algorithm details, limitations, and changes |
| `docs/provenance.md` | Source mapping and validation scope |
| `original/colab_code.txt` | Unmodified source `Code.txt` |
| `tests/` | Pipeline regression tests using synthetic images and stub OCR |

## Method and limitations

The baseline retains the supplied parameters: bilateral filter `(11, 17, 17)`, Canny thresholds `(30, 200)`, ten largest contours, polygon approximation `0.018 × perimeter`, English OCR, and full-image fallback.

A four-sided contour is only a heuristic: bodywork, windows, and signs can be selected. The implementation does not apply aspect-ratio filtering, perspective correction, plate-format validation, or video tracking. Low light, blur, complex backgrounds, oblique plates, and Bangla text remain known limitations. See [methodology](docs/methodology.md).

Supplied vehicle images are available in [samples](samples/). The generated synthetic fixture is not a benchmark. No numerical accuracy, FPS, or benchmark claim is made. A fresh end-to-end OCR evaluation is still required.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests cover crop selection, fallback, coordinate translation, empty OCR results, and invalid image handling. They do not download OCR weights or establish recognition accuracy.

## Future work

Potential extensions from the report include skew correction, low-light enhancement, multilingual OCR, learned plate detection, and video support. These are planned directions, not implemented features.

## Usage and attribution

This repository contains academic project materials and supplied example images. An open-source license has not been selected. Third-party image ownership and redistribution terms have not been verified; confirm permissions before reusing or redistributing those assets. Dependencies retain their own licenses.

## Source materials

- [Project report V3](docs/source/Report_V3.pdf)
- [Presentation V3 (PDF)](docs/source/Presentation_V3.pdf) · [PowerPoint](docs/source/Presentation_V3.pptx)
- [Presentation script](docs/source/Presentation_Script.pdf)
- [Code explanation](docs/source/Code_Explanation.pdf)
- [Original vehicle samples](samples/)

These are supplied academic materials, published with the repository owner's explicit confirmation on 7 November 2026. Historical results have not been independently reproduced.
