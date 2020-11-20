"""Script for feature extraction and matching"""
import numpy as np
import cv2

im1 = cv2.imread('../data/parking/images/img_00000.png', cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread('../data/parking/images/img_00003.png', cv2.IMREAD_GRAYSCALE)
sift_ratio = 0.8

sift = cv2.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(im1, None)
im1_kp = cv2.drawKeypoints(im1, kp_1, None)

kp_2, desc_2 = sift.detectAndCompute(im2, None)
im2_kp = cv2.drawKeypoints(im2, kp_2, None)

concat_img_kp = np.concatenate([im1_kp, im2_kp], axis=1)
cv2.imshow('Img KPs', concat_img_kp)
cv2.waitKey(0)

# Feature Matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc_1, desc_2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < sift_ratio*n.distance:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img_matches = cv2.drawMatchesKnn(im1, kp_1, im2, kp_2, good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('KP Matches', img_matches)
cv2.waitKey(0)