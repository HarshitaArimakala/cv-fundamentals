# Week 2 — Motion Detector

A real-time motion detector built with OpenCV.

## Pipeline
1. **MOG2 background subtraction** — separates foreground from background
2. **Shadow filtering** — removes shadow pixels (value 127) from the mask
3. **Morphological operations** — opening removes noise, closing bridges gaps
4. **Contour detection** — finds distinct moving blobs
5. **Bounding boxes** — draws a labelled box around each detected object with aspect ratio filtering

## Parameters
- `MIN_AREA` — minimum pixel area to count as a real object (default 500)
- `KERNEL_SIZE` — morphological kernel size (default 5×5)
- `history` — MOG2 background model memory in frames (default 500)
- `varThreshold` — MOG2 sensitivity (default 16)