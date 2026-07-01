import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Failed to open camera")
for _ in range(30):
    cap.read()

# --- Capture template from first frame ---
ret, first_frame = cap.read()
if not ret:
    raise SystemExit("Failed to read frame")

# Hardcoded crop — change these values to frame a good region of your object
template = first_frame[150:250, 200:350]
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
th, tw = template_gray.shape[:2]   # template height, width

# Show the cropped template so you can confirm it looks right
cv.imshow("Template", template)
cv.waitKey(500)

# --- Main tracking loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Run template matching
    result = cv.matchTemplate(frame_gray, template_gray, cv.TM_CCOEFF_NORMED)

    # Find best match location
    _, maxVal, _, maxLoc = cv.minMaxLoc(result)

    # Only draw if confidence is sufficient
    if maxVal > 0.8:
        top_left = maxLoc
        bottom_right = (maxLoc[0] + tw, maxLoc[1] + th)
        cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Show confidence score on frame
    cv.putText(frame, f"Confidence: {maxVal:.2f}", (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv.imshow("Template Matching", frame)
    if cv.waitKey(30) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
