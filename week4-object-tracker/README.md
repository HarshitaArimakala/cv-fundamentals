# Lucas-Kanade Sparse Optical Flow

Implementation of sparse optical flow using the Lucas-Kanade method. Detects Shi-Tomasi corners on the first frame, then tracks each one frame-to-frame using `cv2.calcOpticalFlowPyrLK`, drawing motion vectors as the points move.

## Pipeline

* Detect Shi-Tomasi corners on the first frame (`cv2.goodFeaturesToTrack`)
* For each subsequent frame, compute optical flow for those points (`cv2.calcOpticalFlowPyrLK`)
* Filter out points where tracking failed using the returned status array
* Draw a line + circle showing each point's motion from previous → current position
* Update state every frame; refresh corner detection if tracked points drop below 10

## Concepts

* Brightness constancy assumption and the optical flow constraint equation
* Why a single pixel gives an underdetermined system (the aperture problem)
* How the LK window turns this into an overdetermined least-squares solve
* Connection between the LK matrix ($A^T A$) and the Harris structure tensor from Week 3 — why corners are the ideal points to track
* Image pyramids and why they're needed for motion larger than a few pixels

LK tracks corners, not objects. Some drawn vectors will look inconsistent or "random" — this is expected, since background corners and featureless-region points are tracked just as readily as the intended subject, and each genuinely reflects that point's own (sometimes noisy) motion estimate.

# Farneback Dense Optical Flow

Implementation of dense optical flow using the Farneback method. Unlike Day 1's sparse tracker (which follows a chosen set of corners), this computes a velocity vector for every pixel in the frame and visualises the entire motion field as an HSV-coded image.

## Pipeline

* Detect and convert consecutive frames to greyscale.
* Compute the dense flow field (`cv2.calcOpticalFlowFarneback`) — output shape `(H, W, 2)`.
* Convert the flow vectors to magnitude and angle (`cv2.cartToPolar`).
* Build an HSV image: hue = direction, saturation = 255 (fixed), value = normalised magnitude.
* Convert HSV → BGR for display.
* Run as a live loop with both the raw feed and the flow visualisation shown side by side.

## Concepts

* **Polynomial expansion:** Approximating each pixel's local neighbourhood with a quadratic surface instead of comparing raw pixel values directly.
* Why this approach produces a valid (if less certain) flow estimate even on flat, textureless regions — unlike LK, which fails there.
* How a shift in polynomial coefficients between frames is algebraically inverted to recover the displacement (u, v).
* **HSV motion visualisation:** Hue encodes direction, brightness encodes speed.
* NumPy Ellipsis (`...`) indexing for slicing the flow field's channels.

## Sparse vs Dense — when to use which

| Feature | LK (Day 1) | Farneback (Day 2) |
| :--- | :--- | :--- |
| **Tracks** | Chosen corners | Every pixel |
| **Needs texture** | Yes | No |
| **Speed** | Fast | Slower |
| **Best for** | Tracking a specific object | Understanding overall scene motion |
