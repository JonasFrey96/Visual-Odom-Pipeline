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

def extract_features(img):
  # Initiate STAR detector
  orb = cv2.ORB_create()
  # find the keypoints with ORB
  kp = orb.detect(img,None)
  # compute the descriptors with ORB
  kp, des = orb.compute(img, kp)
  # draw only keypoints location,not size and orientation
  img2 = cv2.drawKeypoints(img, kp, img, color=(0,255,0), flags=0)
  return kp, des, img2


class Extractor():
  def __init__(self, cfg=None):
    self._cfg = cfg
    self._lk_params = dict(winSize=(31, 31),
                           maxLevel=3,
                           criteria=(
                           cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    self._feature_params = dict(maxCorners=100,
                                qualityLevel=0.3,
                                minDistance=7,
                                blockSize=11)

    self._sift = cv2.SIFT_create()
    self._descriptor = cv2.SIFT_create()
    self._feature_method = 'sift'
    self._matcher = cv2.BFMatcher()
    self._sift_ratio = 0.80
    self._im_prev = None

  def extend_tracks(self, im_curr, kp, max_bidir_error=10):
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

      k.uv = np.array([x, y]).reshape((2, 1))
      k.t_total += 1
      new_tracks.append(k)

    # Update "previous" image
    self._im_prev = im_curr.copy()
    return new_tracks


  def extract(self, img, t, current_kp=[], detector='sift'):
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
      cv2.circle(mask, (x, y), 5, 0, -1)

    # Detect and describe keypoints
    if detector == 'shi-tomasi':
      kp = cv2.goodFeaturesToTrack(img, mask=mask, **self._feature_params) # Nx1x2
      cv_kp = cv2.KeyPoint_convert(kp)
    elif detector == 'sift':
      cv_kp = self._sift.detect(img, mask=mask)
      kp = cv2.KeyPoint_convert(cv_kp)

    cv_kp, desc = self._descriptor.compute(img, cv_kp)

    # Build output
    return [Keypoint(t_first=t, t_total=1, uv_first=kp[i, :].reshape((2, 1)),
                     uv=kp[i, :].reshape((2, 1)), des=desc[i, :].reshape((-1, 1)))
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
    desc_1 = np.array([pt.des.reshape(1, 128) for pt in list_1]).reshape((len(list_1), -1))
    desc_2 = np.array([pt.des.reshape(1, 128) for pt in list_2]).reshape((len(list_2), -1))
    return self.match(desc_1, desc_2)

  def match_list(self, kp_1, desc_1, keypoints):
    desc_2 = np.array([k.des for k in keypoints])
    kp_2 = [k.uv for k in keypoints]
    return self.match(kp_1, desc_1, kp_2, desc_2)

  def camera_pose(self, K, list_1, list_2, corr='2D-2D', matches=None, T=None):
    """
    Filter list_1 and list_2 using the match list. m.queryIdx for list_1,
    and m.trainIdx for list_2.
    :param K:
    :param list_1:
    :param list_2:
    :param corr:
    :param matches:
    :param T:
    :return:
    """
    if matches:
      list_1_n, list_2_n = [], []
      for match in matches:
        list_1_n.append(list_1[match.queryIdx])
        list_2_n.append(list_2[match.trainIdx])
      list_1 = list_1_n
      list_2 = list_2_n

    if corr == '2D-2D':
      """Get pose from 2D-2D correspondences"""
      # Compute essential matrix
      kp_1_pts = np.array([kp.uv for kp in list_1]).astype(np.float32)
      kp_2_pts = np.array([kp.uv for kp in list_2]).astype(np.float32)
      E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.999, method=cv2.RANSAC, mask=None, threshold=1.0)
      # Remove outliers and recover pose
      inliers = np.nonzero(mask)[0]
      kp_1_pts = kp_1_pts[inliers, :]
      kp_2_pts = kp_2_pts[inliers, :]
      retval, R, t, mask = cv2.recoverPose(E, kp_1_pts, kp_2_pts, K)

    elif corr == '3D-2D':
      """Get pose from 3D-2D correspondences"""
      # Compute essential matrix
      kp_db_pts_3d = np.array([kp.p.T for kp in list_1]).astype(np.float32)
      kp_curr_pts = np.array([kp.uv for kp in list_2]).astype(np.float32)

      kp_db_pts_3d = kp_db_pts_3d.reshape((-1, 1, 3))
      kp_curr_pts = kp_curr_pts.reshape((-1, 1, 2))
      if not isinstance(T, type(None)):
        R, tvec = T[:3, :3], T[:3, 3].reshape((3, 1))
        rvec, _ = cv2.Rodrigues(R)
        retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                      K, None,
                                                      useExtrinsicGuess=True,
                                                      rvec=rvec, tvec=tvec,
                                                      reprojectionError=1.0,
                                                      iterationsCount=1000000,
                                                      confidence=0.95)
      else:
        retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                      K, None, reprojectionError=1.0,
                                                      iterationsCount=1000000,
                                                      confidence=0.95)
      R, _ = cv2.Rodrigues(rvec)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.reshape((3,))
    return inliers.reshape((-1,)).tolist(), H

  def relative_scale(self, K, traj, H1, land_db, land_curr):
    """Re-triangulate landmarks to determine relative scale"""
    land_pts_3d_old = []
    land_pts_3d_new = []
    for i, l in enumerate(land_db):
      t0 = l.t_latest
      H0 = traj._poses[t0]
      converged = self.triangulate(K, H0, H1, [l], [land_curr[i]])
      if len(converged) == 1:
        land_pts_3d_new.append(land_curr[i].p)
        land_pts_3d_old.append(land_db[i].p)

    land_pts_3d_old = np.array(land_pts_3d_old).reshape((-1, 3))
    land_pts_3d_new = np.array(land_pts_3d_new).reshape((-1, 3))

    # Average the ratio of differences
    pair_idxs = combinations(range(land_pts_3d_old.shape[0]), 2)
    ratios = []
    for i, idxs in enumerate(pair_idxs):
      diff_old = np.linalg.norm(land_pts_3d_old[idxs[0], :] - land_pts_3d_old[idxs[1], :])
      diff_new = np.linalg.norm(land_pts_3d_new[idxs[0], :] - land_pts_3d_new[idxs[1], :])
      r = diff_old/diff_new
      if (not np.isinf(r)) and (not np.isnan(r)):
        ratios.append(diff_old/diff_new)

      # TODO: Figure out how many old points to use
      if i > 30:
        break
    return np.mean(np.array(ratios))

  def triangulate_tracks(self, tracked_kp, trajectory, min_track_length=5):
    # TODO: Implement
    return []

  def triangulate(self, K, H0, H1, keyp0, keyp1, t):
    """
    Triangulate keypoints.

    :return: (converged, landmarks)
    converged - indices of triangulated keypoints that converged
    landmarks - resulting 3D landmarks
    """

    uv0 = np.array([kp.uv for kp in keyp0]).astype(np.float32)[:, None, :]
    uv1 = np.array([kp.uv for kp in keyp1]).astype(np.float32)[:, None, :]

    # Construct projection matrices
    P_0 = (K @ H0[:3,:]).astype(np.float32)
    P_1 = (K @ H1[:3,:]).astype(np.float32)
    # points_4D = cv2.triangulatePoints(P_0, P_1, uv0, uv1).reshape((4, -1)).T
    # points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]
    points_3D, converged = triangulatePoints_ILS(P_0, P_1, uv0, uv1)

    landmarks = []
    for i, p in enumerate(points_3D):
      landmarks.append(Landmark(t, p.reshape((3, 1)), keyp1[i].des))

    return np.argwhere(converged == 1).reshape((-1,)), landmarks


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