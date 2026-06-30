import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Webcam is not opening")

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

ret, prev_frame = cap.read()
if not ret:
    exit()

prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

img = prev_frame.copy()
for pt in prev_pts:
    x, y = pt[0]
    cv.circle(img, (int(x), int(y)), 2, (255, 0, 0), 3)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate Optical Flow - LK 
    nextPts, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    good_prev = prev_pts[status== 1]
    good_next = nextPts[status == 1]

    copy = frame.copy()
    for pts in zip(good_prev, good_next):
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        cv.line(copy, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
        cv.circle(copy, (int(x2), int(y2)), 2, (0,0,255), 3)
    
    prev_gray = gray
    prev_pts = good_next.reshape(-1,1,2)
    
    if len(prev_pts) < 10:
        prev_pts = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    
    cv.imshow("Video", copy)     
    if cv.waitKey(30) == ord('q'):  
        break

cap.release()
cv.destroyAllWindows()