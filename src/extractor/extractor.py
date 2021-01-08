from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
from itertools import combinations
from matplotlib import pyplot as plt
from extractor.triangulate import TriangulatorNL
from state.keypoint import Keypoint
from state.landmark import Landmark
import logging

class Extractor():
  def __init__(self, cfg=None, min_kp_dist=10):
    self._cfg = cfg
    self._triangulate_nl = TriangulatorNL(verbosity=0)
    self._lk_params = dict(winSize=(31, 31),
                           maxLevel=3,
                           criteria=(
                           cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    self._shitomasi_params = dict(maxCorners=1000,
                                  qualityLevel=0.03,
                                  minDistance=min_kp_dist,
                                  blockSize=31)

    self._feature_method = 'sift'
    if self._feature_method == 'sift':
      self._features = cv2.SIFT_create(nfeatures=1000)
      self._sift_ratio = 0.80
    elif self._feature_method == 'orb':
      self._features = cv2.ORB_create()
    else:
      exit()
    self._matcher = cv2.BFMatcher()

    self._im_prev = None

  def extend_tracks(self, im_curr, kp, max_bidir_error=30):
    # KLT tracking of candidate keypoints
    new_tracks = []
    if len(kp):
      im0, im1 = self._im_prev, im_curr
      p0 = np.float32([k.uv.T for k in kp]).reshape(-1, 1, 2)
      p1, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p0, None, **self._lk_params)
      p0r, _st, _err = cv2.calcOpticalFlowPyrLK(im0, im1, p1, None, **self._lk_params)
      d = abs(p0 - p0r).reshape(-1, 2).max(-1)
      good = d < max_bidir_error

      for k, (x, y), good_flag in zip(kp, p1.reshape(-1, 2), good):
        if not good_flag:
          continue

        if 0 <= x <= im_curr.shape[1] and 0 <= y <= im_curr.shape[0]:
          k.uv = np.array([x, y]).reshape((2, 1))
          k.t_total += 1
          k.uv_history.append(np.array([x, y]).reshape((2, 1)))
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
    landmarks_dead, kp_dead = [], []

    p1 = p1.reshape((-1, 2)).tolist()
    for i in range(len(landmarks)):
      l, k, (x, y), good_flag = landmarks[i], landmarks_kp[i], p1[i], good[i]
      if (not good_flag) or not (0 <= x <= im_curr.shape[1] and 0 <= y <= im_curr.shape[0]):
        landmarks_dead.append(l)
        kp_dead.append(k)
        continue
      else:
        k.uv = np.array([x, y]).reshape((2, 1))
        k.t_total += 1
        k.uv_history.append(np.array([x, y]).reshape((2, 1)))
        l.t_latest += 1

        kp_new.append(deepcopy(k))
        landmarks_new.append(l)

    return landmarks_new, kp_new, landmarks_dead, kp_dead

  def extract(self, img, t, current_kp=[], detector='custom', mask_radius=5,
              describe=False):
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

      # TODO: filter out keypoints with bad cornerness

    if not isinstance(kp, type(None)):
      if describe:
        cv_kp, desc = self._features.compute(img, cv_kp)
        kp = cv2.KeyPoint_convert(cv_kp)
      else:
        desc = np.zeros((kp.shape[0], 1))

      # Build output
      return [Keypoint(t_first=t, t_total=1, uv_first=kp[i, :].reshape((2, 1)),
                       uv=kp[i, :].reshape((2, 1)), des=desc[i, :].reshape((-1, 1)),
                       uv_history=[kp[i, :].reshape((2, 1))])
              for i in range(len(kp))]
    else:
      return []

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
                                                    confidence=0.9999)
      R, _ = cv2.Rodrigues(rvec)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.reshape((3,))
    return inliers.reshape((-1,)).tolist(), H

  def triangulate_tracks(self, K, candidates_kp, trajectory, t_curr, refine=True,
                         min_track_length=5, min_bearing_angle=10, max_err_reproj=4.0):
    """Triangulate tracks that exceed the minimum track length.
        Only create a new landmark if bearing angle exceeds the threshold.
        Non-linearly refine points and keypoints if flag is set.
        Remove tracked keypoints that get triangulated (whether succ. or not)."""
    landmarks_new, landmarks_kp_new = [], []

    # Split keypoint tracks based on track length threshold
    landmarks_kp_tmp = [kp for kp in candidates_kp if kp.t_total >= min_track_length] # Kp's we try to triangulate
    candidates_kp_new = [kp for kp in candidates_kp if kp.t_total < min_track_length]

    if len(landmarks_kp_tmp) > 0:
      logging.info(f"Non-linearly triangulating {len(landmarks_kp_tmp)} keypoint pairs")
      H1 = trajectory[len(trajectory) - 1]

      # Triangulate keypoint tracks in groups based on their t_first
      t_first_groups = set([k.t_first for k in landmarks_kp_tmp])
      for t_first in t_first_groups:
        kp_1 = [kp for kp in landmarks_kp_tmp
                if kp.t_first == t_first]

        H0 = trajectory[t_first]
        kp_0 = deepcopy(kp_1)
        for kp in kp_0:
          kp.uv = kp.uv_first

        l, kp_0, kp_1 = self.triangulate_nonlinear(K, H0, H1, kp_0, kp_1, t_curr, max_err_reproj=max_err_reproj)

        if len(l):
            # # Update landmark kp's - remove history of the kp track, since there was no landmar
            # for kp in kp_1:
            #   kp.uv = kp.uv_first
            #   kp.uv_history = [deepcopy(kp.uv)]

            # Compute bearing angle: theta = cosinv((b^2 + c^2 - a^2) / (2bc))
            # a = baseline
            # b, c = dist to landmark at t0 and t1
            Hrel = H1 @ np.linalg.inv(H0)
            P_homo = np.concatenate([l[0].p, np.zeros((1, 1))], axis=0).reshape((4, 1))
            a = np.linalg.norm(Hrel)
            b = np.linalg.norm(H0 @ P_homo)
            c = np.linalg.norm(H1 @ P_homo)
            bearing_angle = np.rad2deg(np.arccos((b*b + c*c - a*a)/(2*b*c)))

            if (not np.isnan(bearing_angle)) and (bearing_angle > min_bearing_angle):
              landmarks_new += l
              landmarks_kp_new += kp_1

    return landmarks_new, landmarks_kp_new, candidates_kp_new

  def triangulate_nonlinear(self, K, H0, H1, keyp0, keyp1, t, max_err_reproj=1.0):
    """
    Triangulate keypoints and non-linearly refine the results
    (triangulated points as well as the original keypoints).

    :param t: Needed for landmark creation
    :return: landmarks, keyp0, keyp1 (Newly created landmarks, modified keypoints)
    """
    landmarks = self.triangulate(K, H0, H1, keyp0, keyp1, t)
    return self._triangulate_nl.refine(K, landmarks, H0, H1, keyp0, keyp1, max_err_reproj) # landmarks, keyp0, keyp1

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
    P_0 = (K @ H0[:3, :]).astype(np.float32)
    P_1 = (K @ H1[:3, :]).astype(np.float32)
    points_4D = cv2.triangulatePoints(P_0, P_1, uv0, uv1).reshape((4, -1)).T
    points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]

    landmarks = []
    for i, p in enumerate(points_3D.tolist()):
      landmarks.append(Landmark(t, np.array(p).reshape((3, 1)), keyp1[i].des))

    return landmarks


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