import numpy as np 
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
from itertools import combinations
from matplotlib import pyplot as plt
from extractor.triangulate import triangulatePoints_ILS
from state.keypoint import Keypoint
from state.landmark import Landmark

class Extractor():
  def __init__(self, cfg=None):
    self._cfg = cfg
    self._lk_params = dict(winSize=(31, 31),
                           maxLevel=3,
                           criteria=(
                           cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    self._shitomasi_params = dict(maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=5,
                                blockSize=31)

    self._ba_window_size = 999 #
    self._feature_method = 'sift'
    if self._feature_method == 'sift':
      self._features = cv2.SIFT_create()
      self._sift_ratio = 0.80
    elif self._feature_method == 'orb':
      self._features = cv2.ORB_create()
    else:
      exit()
    self._matcher = cv2.BFMatcher()

    self._im_prev = None

  def extend_tracks(self, im_curr, kp, max_bidir_error=30):
    # KLT tracking of keypoints
    im0, im1 = self._im_prev, im_curr
    p0 = np.float32([k.uv for k in kp]).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self._lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p1, None, **self._lk_params)
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    good = d < max_bidir_error
    new_tracks = []
    for k, (x, y), good_flag in zip(kp, p1.reshape(-1, 2), good):
      if not good_flag:
        continue

      if 0 <= x <= im_curr.shape[1] and 0 <= y <= im_curr.shape[0]:
        k.uv = np.array([x, y]).reshape((2, 1))
        k.t_total += 1
        k.uv_history.append(np.array([x, y]).reshape((2, 1)))
        if len(k.uv_history) > self._ba_window_size:
          del k.uv_history[0]
        new_tracks.append(k)

    return new_tracks

  def extend_landmarks(self, im_curr, landmarks, landmarks_kp, max_bidir_error=30):
    # KLT tracking of keypoints
    im0, im1 = self._im_prev, im_curr
    p0 = np.float32([k.uv for k in landmarks_kp]).reshape(-1, 1, 2)
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self._lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p1, None, **self._lk_params)
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    good = d < max_bidir_error
    landmarks_new, kp_new = [], []

    p1 = p1.reshape((-1, 2)).tolist()
    for i in range(len(landmarks)):
      l, k, (x, y), good_flag = landmarks[i], landmarks_kp[i], p1[i], good[i]
      if not good_flag:
        continue

      if 0 <= x <= im_curr.shape[1] and 0 <= y <= im_curr.shape[0]:
        k.uv = np.array([x, y]).reshape((2, 1))
        k.t_total += 1
        k.uv_history.append(np.array([x, y]).reshape((2, 1)))

        if len(k.uv_history) > self._ba_window_size:
          del k.uv_history[0]
        kp_new.append(deepcopy(k))
        landmarks_new.append(l)

    return landmarks_new, kp_new

  def extract(self, img, t, current_kp=[], detector='custom', mask_radius=5):
    """
    Given a grayscale image, detect keypoints and generate
    descriptors for each of them. Return a list of custom Keypoint objects.

    This function is mainly used to initialize feature tracks.
    :param img: grayscale image
    :param t: current t_step used to construct the Keypoint objects
    :param current_kp: list of currently tracked Keypoint objects
    :return: list of Keypoints
    """
    # Create a mask to not re-detect currently tracked kps
    img = img.copy()
    mask = np.zeros_like(img)
    mask[:] = 255
    for x, y in [np.int32(kp.uv) for kp in current_kp]:
      cv2.circle(mask, (x, y), mask_radius, 0, -1)

    # Detect and describe keypoints
    if detector == 'shi-tomasi':
      kp = cv2.goodFeaturesToTrack(img, mask=mask, **self._shitomasi_params)
      cv_kp = cv2.KeyPoint_convert(kp)
    elif detector == 'custom':
      cv_kp = self._features.detect(img, mask=mask)
      kp = cv2.KeyPoint_convert(cv_kp)

    cv_kp, desc = self._features.compute(img, cv_kp)
    kp = cv2.KeyPoint_convert(cv_kp)

    # Build output
    return [Keypoint(t_first=t, t_total=1, uv_first=kp[i, :].reshape((2, 1)),
                     uv=kp[i, :].reshape((2, 1)), des=desc[i, :].reshape((-1, 1)),
                     uv_history=[kp[i, :].reshape((2, 1))])
            for i in range(len(kp))]

  def match(self, desc_1, desc_2):
    if self._feature_method == 'sift':
      matches = self._matcher.knnMatch(desc_1, desc_2, k=2)
      good = []

      for m, n in matches:
          if m.distance < self._sift_ratio*n.distance:
            good.append(m)
      return good

    elif self._feature_method == 'orb':
      return self._matcher.match(desc_1, desc_2)

  def match_lists(self, list_1, list_2):
    """Match two lists of keypoints/landmarks based on their descriptors.
    Returns a list of matches. Each match has m.queryIdx for list_1,
    and m.trainIdx for list_2."""
    desc_dim = len(list_1[0].des)
    desc_1 = np.array([pt.des.reshape(1, desc_dim) for pt in list_1]).reshape((len(list_1), -1))
    desc_2 = np.array([pt.des.reshape(1, desc_dim) for pt in list_2]).reshape((len(list_2), -1))
    return self.match(desc_1, desc_2)

  def match_list(self, kp_1, desc_1, keypoints):
    desc_2 = np.array([k.des for k in keypoints])
    kp_2 = [k.uv for k in keypoints]
    return self.match(kp_1, desc_1, kp_2, desc_2)

  def camera_pose(self, K, list_1, list_2, corr='2D-2D', max_err_reproj=4.0):
    if corr == '2D-2D':
      """Get pose from 2D-2D correspondences"""
      # Compute essential matrix
      kp_1_pts = np.array([kp.uv.T for kp in list_1]).astype(np.float32)
      kp_2_pts = np.array([kp.uv.T for kp in list_2]).astype(np.float32)
      E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0)
      # Remove outliers and recover pose
      inliers = np.nonzero(mask)[0]
      kp_1_pts = kp_1_pts[inliers, :]
      kp_2_pts = kp_2_pts[inliers, :]
      retval, R, t, mask = cv2.recoverPose(E, kp_1_pts, kp_2_pts, K)

    elif corr == '3D-2D':
      """Get pose from 3D-2D correspondences"""
      kp_db_pts_3d = np.array([kp.p.T for kp in list_1]).astype(np.float32)
      kp_curr_pts = np.array([kp.uv.T for kp in list_2]).astype(np.float32)

      kp_db_pts_3d = kp_db_pts_3d.reshape((-1, 1, 3))
      kp_curr_pts = kp_curr_pts.reshape((-1, 1, 2))

      retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                    K, None, reprojectionError=max_err_reproj,
                                                    iterationsCount=1000000,
                                                    confidence=0.99,
                                                    flags=cv2.SOLVEPNP_P3P)
      R, _ = cv2.Rodrigues(rvec)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.reshape((3,))
    return inliers.reshape((-1,)).tolist(), H

  def triangulate_tracks(self, K, candidates_kp, trajectory, t_curr, min_track_length=5, min_bearing_angle=10, max_err_reproj=4.0):
    """Triangulate tracks that exceed the minimum track length.
       Only create a new landmark if bearing angle exceeds the threshold.

     Remove tracked keypoints that get triangulated (whether succ. or not).
    """
    landmarks_new = []

    # Split keypoint tracks based on track length threshold
    landmarks_kp_new = [kp for kp in candidates_kp if kp.t_total >= min_track_length]
    candidates_kp_new = [kp for kp in candidates_kp if kp.t_total < min_track_length]

    if len(landmarks_kp_new) > 0:

      # Triangulate keypoint tracks independently
      H1 = trajectory[len(trajectory)-1]
      for kp_1 in landmarks_kp_new:
        kp_0 = Keypoint(kp_1.t_first, kp_1.t_total, kp_1.uv_first,
                        kp_1.uv_first, kp_1.des, [kp_1.uv_first])
        H0 = trajectory[kp_0.t_first]
        converged, l = self.triangulate(K, H0, H1, [kp_0], [kp_1], t_curr)
        if len(converged):
          # Compute bearing angle: theta = cosinv((b^2 + c^2 - a^2) / (2bc))
          # a = baseline
          # b, c = dist to landmark at t0 and t1
          Hrel = H1 @ np.linalg.inv(H0)
          P_homo = np.concatenate([l[0].p, np.zeros((1, 1))], axis=0).reshape((4, 1))
          a = np.linalg.norm(Hrel)
          b = np.linalg.norm(H0 @ P_homo)
          c = np.linalg.norm(H1 @ P_homo)
          bearing_angle = np.rad2deg(np.arccos((b*b + c*c - a*a)/(2*b*c)))

          # Compute reprojection error (Mean reprojection error btwn t0 and t1)
          px_0 = K @ (H0 @ P_homo)[:3, 0:1]
          px_0 = (px_0 / px_0[2:3, 0:1])[:2, 0:1]
          err_0 = np.linalg.norm(kp_0.uv_first - px_0)
          px_1 = K @ (H1 @ P_homo)[:3, 0:1]
          px_1 = (px_1 / px_1[2:3, 0:1])[:2, 0:1]
          err_1 = np.linalg.norm(kp_1.uv_first - px_1)
          # err_reproj = np.mean([err_0, err_1])
          err_reproj = 0

          if (not np.isnan(bearing_angle)) and (bearing_angle > min_bearing_angle) \
             and (err_reproj < max_err_reproj):
            landmarks_new += l

    return landmarks_new, landmarks_kp_new, candidates_kp_new

  def triangulate(self, K, H0, H1, keyp0, keyp1, t):
    """
    Triangulate keypoints.

    :return: (converged, landmarks)
    converged - indices of triangulated keypoints that converged
    landmarks - resulting 3D landmarks
    """

    uv0 = np.array([kp.uv.T for kp in keyp0]).astype(np.float32).reshape((-1, 1, 2))
    uv1 = np.array([kp.uv.T for kp in keyp1]).astype(np.float32).reshape((-1, 1, 2))

    # Construct projection matrices
    P_0 = (K @ H0[:3,:]).astype(np.float32)
    P_1 = (K @ H1[:3,:]).astype(np.float32)
    points_4D = cv2.triangulatePoints(P_0, P_1, uv0, uv1).reshape((4, -1)).T
    points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]
    converged = list(range(points_3D.shape[0]))
    # points_3D, converged = triangulatePoints_ILS(P_0, P_1, uv0, uv1)
    # converged = np.argwhere(converged == 1).reshape((-1,))

    landmarks = []
    for i, p in enumerate(points_3D.tolist()):
      landmarks.append(Landmark(t, np.array(p).reshape((3, 1)), keyp1[i].des))

    return converged, landmarks


  def reprojection_error(self, landmarks_3D, landmarks_2D, K, norm='L1', T=np.eye(4)):
    if norm not in ['L2', 'L1']:
      print(f"Invalid norm specified")
      exit()

    err = 0.0
    N = len(landmarks_2D)
    for i in range(N):
      l_2D, l_3D = landmarks_2D[i], landmarks_3D[i]
      l_3D.p = (T @ np.concatenate([l_3D.p.reshape((3, 1)), np.ones((1, 1))]))[:3, :]
      uv_homo = K @ l_3D.p
      uv = (uv_homo[:2] / uv_homo[2]).astype(np.int).reshape((2,))
      uv_kpt = l_2D.uv.astype(np.int).reshape((2,))
      diff = np.linalg.norm(uv - uv_kpt)

      if norm == 'L1':
        err += diff
      elif norm == 'L2':
        err += (diff*diff)
    return err/N

if __name__ == "__main__":
  pass