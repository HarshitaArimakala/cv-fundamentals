import cv2
import numpy as np

# ── SETUP ──────────────────────────────────────────────
cap = cv2.VideoCapture(0)       # 0 = default webcam

# HSV bounds for orange (tune these for your target color)
lower = np.array([10, 100, 100])
upper = np.array([25, 255, 255])

# Morphology kernel — a 5x5 block of ones
kernel = np.ones((5, 5), np.uint8)

# ── MAIN LOOP ──────────────────────────────────────────
while True:

    # grab a frame — ret is False if camera disconnected
    ret, frame = cap.read()
    if not ret:
        break

    # BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # threshold → binary mask
    mask = cv2.inRange(hsv, lower, upper)

    # Optional: clean the mask
    # OPEN  → removes small white noise specs outside the object
    # CLOSE → fills small black holes inside the object
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # apply mask
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display all three panels side by side
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display  = np.hstack([frame, mask_3ch, result])
    cv2.imshow("Original | Mask | Result", display)

    # Wait 1ms per frame — quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── CLEANUP ────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()