import cv2 as cv
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_AREA     = 500
KERNEL_SIZE  = (5, 5)
KERNEL_SHAPE = cv.MORPH_ELLIPSE

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Camera not opening")
    exit()

# ── MOG2 + structuring element ─────────────────────────────────────────────────
bgSub = cv.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)
kernel = cv.getStructuringElement(KERNEL_SHAPE, KERNEL_SIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Unable to receive frame")
        break

    # ── Steps 1–3: MOG2 → shadow filter → morphology ──────────────────────────
    fgMask     = bgSub.apply(frame, learningRate=-1)
    _, clean_mask = cv.threshold(fgMask, 200, 255, cv.THRESH_BINARY)
    opening    = cv.morphologyEx(clean_mask, cv.MORPH_OPEN,  kernel)
    combined   = cv.morphologyEx(opening,    cv.MORPH_CLOSE, kernel)

    # ── Step 4: global motion check ────────────────────────────────────────────
    S      = np.count_nonzero(combined)
    motion = S > MIN_AREA

    # ── Step 5: find all blob boundaries ──────────────────────────────────────
    # combined must be CV_8UC1 (single-channel binary) — not the BGR frame
    contours, _ = cv.findContours(combined, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # ── Step 6: filter, draw, count ───────────────────────────────────────────
    display = frame.copy()
    count   = 0   # reset every frame

    for contour in contours:

        # area filter — reject noise blobs
        area = cv.contourArea(contour)
        if area < MIN_AREA:
            continue

        # bounding rect
        x, y, w, h = cv.boundingRect(contour)

        # aspect ratio filter — reject geometric noise (shadow streaks, edge vibrations)
        aspect_ratio = float(w) / h
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue

        # passed both filters — draw and count
        count += 1
        cv.drawContours(display, [contour], -1, (0, 255, 0), 2)
        cv.rectangle(display,  (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(display, "Motion", (x, max(y - 10, 10)), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

    # ── Step 7: global labels ──────────────────────────────────────────────────
    status_label = f"Motion: {S}px" if motion else f"Still: {S}px"
    status_color = (0, 0, 200)       if motion else (0, 200, 0)
    cv.putText(display, status_label,       (12, 34), cv.FONT_HERSHEY_COMPLEX, 0.8, status_color, 2)
    cv.putText(display, f"Objects: {count}", (12, 65), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

    cv.imshow("Camera",   display)
    cv.imshow("Combined", combined)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()