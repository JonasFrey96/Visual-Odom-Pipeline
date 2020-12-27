"""Script for KLT Tracking"""
import cv2
import numpy as np
from glob import glob
import os

# im_dir = '/media/carter/Samsung_T5/vo_data/parking/images'
im_dir = '/media/carter/Samsung_T5/vo_data/kitti/00/image_0'
K = [[331.37, 0.0,       320.0],
     [0.0,      369.568, 240.0],
     [0.0,      0.0,       1.0]]
im_paths = sorted(glob(os.path.join(im_dir, '*.png')))
frame_idx = 0
detect_interval = 1
max_tracks = 20
max_bidir_error = 30

lk_params = dict( winSize  = (31, 31),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 10,
                       blockSize = 11 )

tracks = []
while frame_idx < len(im_paths):

    curr_frame = cv2.imread(im_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    # curr_frame = cv2.blur(curr_frame, (3, 3))
    curr_frame = cv2.bilateralFilter(curr_frame, 3, 1, 1)
    vis = curr_frame.copy()
    # curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if len(tracks) > 0:
        im0, im1 = prev_frame, curr_frame
        p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p1, None, **lk_params)
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        good = d < max_bidir_error
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue

            tr.append((x, y))
            if len(tr) > max_tracks:
                del tr[0]

            new_tracks.append(tr)
            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

        tracks = new_tracks
        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

    if len(tracks) == 0:
        cv2.waitKey(0)

    print(f"Tracked points: {len(tracks)}")
    # curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = curr_frame.copy()
    if frame_idx % detect_interval == 0:
        mask = np.zeros_like(curr_frame_gray)
        mask[:] = 255
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # p = cv2.goodFeaturesToTrack(curr_frame_gray, mask=mask, **feature_params)
        # cv_p = cv2.KeyPoint_convert(p)
        sift = cv2.SIFT_create()
        cv_p = sift.detect(curr_frame_gray, mask=mask)
        cv_p, d = sift.compute(curr_frame_gray, cv_p)
        p = cv2.KeyPoint_convert(cv_p)
        if p is not None:
            for x, y in np.float32(p).reshape(-1, 2):
                tracks.append([(x, y)])

    frame_idx += 1
    prev_frame = curr_frame
    cv2.imshow("Tracking", vis)
    cv2.waitKey(1)
