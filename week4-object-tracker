import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam is not opening")

# Read first frame
ret, frame1 = cap.read()
if not ret:
    raise SystemExit("Failed to read initial frame")
prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

while True:
    # Read second consecutive frame
    ret, frame2 = cap.read()
    if not ret:
        break
    
    gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow  
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                    pyr_scale=0.5, levels=3, winsize=15,
                                    iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

    # Convert to magnitude and angle
    u = flow[..., 0]
    v = flow[..., 1]
    mag, angle = cv.cartToPolar(u, v)

    # HSV visualisation
    hsv = np.zeros_like(frame2)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    # Display both windows
    cv.imshow("Raw Frame", frame2)
    cv.imshow("Dense Flow HSV", bgr)
    
    # Update previous frame for the next iteration
    prev_gray = gray
    
    if cv.waitKey(30) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()