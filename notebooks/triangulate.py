"""Script for triangulating correspondences"""
import cv2
import numpy as np

"""Script for estimating the fundamental matrix between two frames"""
import cv2
import numpy as np
from random import randint

im1 = cv2.imread('../data/parking/images/img_00000.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/parking/images/img_00003.png', cv2.IMREAD_GRAYSCALE)
sift_ratio = 0.75

sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(im1, None)
im1_kp = cv2.drawKeypoints(im1, kp_1, None)

kp_2, desc_2 = sift.detectAndCompute(im2, None)
im2_kp = cv2.drawKeypoints(im2, kp_2, None)

# Feature Matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_1, desc_2, k=2)

# Apply ratio test
good = []
kp_1_matches, kp_2_matches = [], []
for m, n in matches:
    if m.distance < sift_ratio*n.distance:
        good.append([m])
        kp_1_matches.append(kp_1[m.queryIdx])
        kp_2_matches.append(kp_2[m.trainIdx])

# cv.drawMatchesKnn expects list of lists as matches.
img_matches = cv2.drawMatchesKnn(im1, kp_1, im2, kp_2, good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('KP Matches', img_matches)
cv2.waitKey(0)

# Compute essential matrix
kp_1_pts = np.array([kpt.pt for kpt in kp_1_matches]).astype(np.float)
kp_2_pts = np.array([kpt.pt for kpt in kp_2_matches]).astype(np.float)
K = [[331.37, 0.0,       320.0],
     [0.0,      369.568, 240.0],
     [0.0,      0.0,       1.0]]
K = np.array(K)
E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.999, method=cv2.RANSAC, mask=None, threshold=1)
mask = np.nonzero(mask)[0]

# Remove outliers and recover pose
kp_1_pts = kp_1_pts[mask, :]
kp_2_pts = kp_2_pts[mask, :]
retval, R, t, mask = cv2.recoverPose(E, kp_1_pts, kp_2_pts, K)

# Construct projection matrices
T_1 = np.eye(3, 4)
P_1 = K @ T_1

T_2 = np.eye(3, 4)
T_2[:3, :3] = R
T_2[:3, 3] = t.reshape((3,))
P_2 = K @ T_2

points_4D = cv2.triangulatePoints(P_1, P_2, kp_1_pts.T, kp_2_pts.T).reshape((4, -1)).T
points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]

# Reproject and check
uv_1_homo = (K @ points_3D.T).T
uv_1 = (uv_1_homo/uv_1_homo[:, 2:3]).astype(np.int)
uv_2_homo = (P_2 @ points_4D.T).T
uv_2 = (uv_2_homo/uv_2_homo[:, 2:3]).astype(np.int)

im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2BGR)
im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR)
for i in range(uv_1.shape[0]):
    color = tuple([randint(0, 256) for x in range(3)])
    cv2.circle(im1, (uv_1[i, :][0], uv_1[i, :][1]), 2, color, -1)
    cv2.circle(im2, (uv_2[i, :][0], uv_2[i, :][1]), 2, color, -1)

cv2.imshow("Re-proj 1", im1)
cv2.imshow("Re-proj 2", im2)
cv2.waitKey(0)
