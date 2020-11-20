"""Script for solvePnP"""
import cv2
import numpy as np

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

# Compute essential matrix
kp_1_pts = np.array([kpt.pt for kpt in kp_1_matches]).astype(np.float)
kp_2_pts = np.array([kpt.pt for kpt in kp_2_matches]).astype(np.float)
K = [[331.37, 0.0,       320.0],
     [0.0,      369.568, 240.0],
     [0.0,      0.0,       1.0]]
K = np.array(K)
E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.999, method=cv2.RANSAC, mask=None, threshold=1)

# Remove outliers and recover pose
kp_1_pts, desc_1 = kp_1_pts[mask, :], desc_1[mask, :].reshape((-1, 128))
kp_2_pts, desc_2 = kp_2_pts[mask, :], desc_2[mask, :].reshape((-1, 128))
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

# Use SolvePnP to get the pose of a new frame
im3 = cv2.imread('../data/parking/images/img_00007.png', cv2.IMREAD_GRAYSCALE)
kp_3, desc_3 = sift.detectAndCompute(im3, None)

# Match new features to old ones
matches = bf.knnMatch(desc_2, desc_3, k=2)

# Apply ratio test to filter matches, Construct 3D-2D Correspondences
good = []
corr_2D, corr_3D = [], []
kp_2_matches, kp_3_matches = [], []
for m, n in matches:
    if m.distance < sift_ratio*n.distance:
        good.append([m])
        kp_2_matches.append(kp_2[m.queryIdx])
        kp_3_matches.append(kp_3[m.trainIdx])
        corr_3D.append(points_3D[m.queryIdx, :])
        corr_2D.append(kp_3[m.trainIdx].pt)

corr_3D = np.array(corr_3D).reshape((-1, 1, 3))
corr_2D = np.array(corr_2D).reshape((-1, 1, 2))
retval, rvec, tvec, inliers = cv2.solvePnPRansac(corr_3D, corr_2D, K, None)
