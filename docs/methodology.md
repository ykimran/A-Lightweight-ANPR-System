# Methodology and implementation notes

1. Read a vehicle image as BGR using OpenCV.
2. Convert to grayscale and apply a bilateral filter with diameter 11 and sigma values 17 and 17.
3. Run Canny with thresholds 30 and 200.
4. Retrieve contours with `RETR_TREE` and `CHAIN_APPROX_SIMPLE`, retaining the ten largest by area.
5. Approximate each contour at 0.018 times its perimeter. Select the first nondegenerate polygon with four vertices.
6. Crop the bounding rectangle from the filtered grayscale image. Save a masked visualization.
7. If no contour qualifies, use the full filtered image.
8. Run EasyOCR with English enabled, combine detections in returned order, and uppercase text.
9. Translate each crop-relative OCR polygon by the crop origin before drawing it on the original image.

## Changes from the supplied Code.txt

The original file remains unmodified under `original/`. The new implementation removes the Colab-only upload dependency from the reusable core, adds a CLI and JSON output, reuses the reader when processing notebook uploads, and saves stage images. It fixes crop-relative OCR box placement and the original loop's final-label variable reuse. Grayscale crops are displayed as grayscale rather than passed through a BGR conversion. Uppercase output follows the report appendix. Zero-area four-point polygons are rejected.

The supplied report mentions plate size/aspect-ratio constraints, but the source code does not implement those checks. This repository preserves the actual contour heuristic rather than claiming that those checks exist.

## Evaluation limits

The report provides qualitative examples, not a labeled quantitative benchmark. Sample success/error labels are inherited from the source archive. No numeric accuracy or speed measurement has been reconstructed. Test images do not establish robustness in live traffic. Full-image fallback may read unrelated text and is not proof of successful plate localization.

Bangla OCR, perspective correction, training, real-time video, and automatic plate-format validation are not implemented. CPU operation is supported by the CLI default, but EasyOCR's pretrained neural models still require memory and an initial weight download.
