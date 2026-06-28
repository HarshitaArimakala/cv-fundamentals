## Week 3 Capstone - Document Scanner

A classical CV pipeline that detects a document in an image, extracts it, 
and produces a clean top-down scan — similar to CamScanner.

### Pipeline
1. Convert to grayscale and apply Gaussian blur
2. Canny edge detection to find document edges
3. Contour detection to locate the document boundary
4. Identify the four corner points of the document
5. Perspective transform (warpPerspective) to get a flat top-down view

### Concepts Used
- Edge detection (Canny)
- Contour detection and approximation
- Perspective transform / homography
- `cv2.getPerspectiveTransform` and `cv2.warpPerspective`

### How to Run
```bash
python code.py
```

### Input / Output
- Input: Image containing a document (phone photo, angled scan)
- Output: Flat, top-down corrected version of the document

### Key Takeaway
Homography maps four points in one plane to four points in another — 
the core idea behind document rectification, AR overlays, and camera calibration.