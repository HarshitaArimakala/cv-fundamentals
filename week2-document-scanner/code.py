import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Images
img = cv.imread("Box.png", cv.IMREAD_GRAYSCALE)
img_rotated = cv.imread("Box_Rotated.png", cv.IMREAD_GRAYSCALE)

if img is None or img_rotated is None:
    print("Image not loaded — check file path")
    exit()

# ORB
orb = cv.ORB_create(nfeatures=10000)
kpA, desA = orb.detectAndCompute(img, None)
kpB, desB = orb.detectAndCompute(img_rotated, None)
print("Keypoints A:", len(kpA))
print("Keypoints B:", len(kpB))

# FLANN Matcher
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desA, desB, k=2)

# Ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
print("Good matches:", len(good))

if len(good) < 10:
    print("Not enough good matches:", len(good))
    exit()

# Extract coordinates
pts_A = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts_B = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

# Homography
H, mask = cv.findHomography(pts_A, pts_B, cv.RANSAC, 5.0)

if H is None or np.sum(mask) < 10:
    print("Homography unreliable — inliers:", np.sum(mask) if mask is not None else 0)
else:
    print("Inliers:", np.sum(mask))

    # Bounding quad
    h, w = img.shape
    corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
    corners_transformed = cv.perspectiveTransform(corners, H)

    img_rotated_color = cv.cvtColor(img_rotated, cv.COLOR_GRAY2BGR)
    cv.polylines(img_rotated_color, [np.int32(corners_transformed)], True, (0,255,0), 3)

    # Inlier matches only
    inlier_matches = [m for m, keep in zip(good, mask.ravel()) if keep]
    match_img = cv.drawMatchesKnn(img, kpA, img_rotated, kpB,
                                [[m] for m in inlier_matches], None,
                                flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(cv.cvtColor(img_rotated_color, cv.COLOR_BGR2RGB))
    plt.title("Bounding Quad")

    plt.subplot(1,2,2)
    plt.imshow(match_img)
    plt.title("Inlier Matches")

    plt.tight_layout()
    plt.show()