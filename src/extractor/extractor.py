import numpy as np 
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
from itertools import combinations
from matplotlib import pyplot as plt
from extractor.triangulate import triangulatePoints_ILS

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
  def __init__(self,cfg=None):
    self._cfg = cfg
    self._feature_method = 'sift'
    if self._feature_method == 'orb':
      self._detector = cv2.ORB_create()
      self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif self._feature_method == 'sift':
      self._detector = cv2.SIFT_create()
      self._matcher = cv2.BFMatcher()
      self._sift_ratio = 0.80


  def extract(self, img):

    kp_1, desc_1 = self._detector.detectAndCompute(img, None)
    return kp_1, desc_1

  def match(self, kp_1, desc_1, kp_2, desc_2):
    if self._feature_method == 'sift':
      matches = self._matcher.knnMatch(desc_1, desc_2, k=2)
      good = []

      for m, n in matches:
          if m.distance < self._sift_ratio*n.distance:
            good.append(m)
      return good

    elif self._feature_method == 'orb':
      return self._matcher.match(desc_1, desc_2)

  def match_list(self, kp_1, desc_1, keypoints):
    desc_2 = np.array([k.des for k in keypoints])
    kp_2 = [k.uv for k in keypoints]
    return self.match(kp_1, desc_1, kp_2, desc_2)

  def camera_pose(self, K, land1, land2, corr='2D-2D', T=None):
    if corr == '2D-2D':
      """Get pose from 2D-2D correspondences"""
      # Compute essential matrix
      kp_1_pts = np.array([kp.uv for kp in land1]).astype(np.float32)
      kp_2_pts = np.array([kp.uv for kp in land2]).astype(np.float32)
      E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.9999, method=cv2.RANSAC, mask=None, threshold=1.0)
      # Remove outliers and recover pose
      inliers = np.nonzero(mask)[0]
      kp_1_pts = kp_1_pts[inliers, :]
      kp_2_pts = kp_2_pts[inliers, :]
      retval, R, t, mask = cv2.recoverPose(E, kp_1_pts, kp_2_pts, K)

    elif corr == '3D-2D':
      """Get pose from 3D-2D correspondences"""
      # Compute essential matrix
      kp_db_pts_3d = np.array([kp.p.T for kp in land1]).astype(np.float32)
      kp_curr_pts = np.array([kp.uv for kp in land2]).astype(np.float32)

      kp_db_pts_3d = kp_db_pts_3d.reshape((-1, 1, 3))
      kp_curr_pts = kp_curr_pts.reshape((-1, 1, 2))
      if not isinstance(T, type(None)):
        R, tvec = T[:3, :3], T[:3, 3].reshape((3, 1))
        rvec, _ = cv2.Rodrigues(R)
        retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                      K, None,
                                                      useExtrinsicGuess=True,
                                                      rvec=rvec, tvec=tvec,
                                                      reprojectionError=2.0,
                                                      iterationsCount=100000,
                                                      confidence=0.95)
      else:
        retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                      K, None, reprojectionError=2.0,
                                                      iterationsCount=100000,
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


  def triangulate(self, K, H0, H1, keyp0, keyp1):
    """Triangulate a batch of keypoints between two images
     keypoints are passed by reference.
    Results will be stored in keyp1.p (Triangulated points are expressed
    in the coordinate frame of camera 1).

    Parameters
    ----------
    K : [type]
        [description]
    H0 : [type]
        [description]
    H1 : [type]
        [description]
    keyp0 : [type]
        [description]
    keyp1 : [type]
        [description]
    """
    uv0 = np.array([kp.uv for kp in keyp0]).astype(np.float32)[:, None, :]
    uv1 = np.array([kp.uv for kp in keyp1]).astype(np.float32)[:, None, :]

    # Construct projection matrices
    P_0 = (K @ H0[:3,:]).astype(np.float32)
    P_1 = (K @ H1[:3,:]).astype(np.float32)
    # points_4D = cv2.triangulatePoints(P_0, P_1, uv0, uv1).reshape((4, -1)).T
    # points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]
    points_3D, converged = triangulatePoints_ILS(P_0, P_1, uv0, uv1)

    for j, k in enumerate(keyp1):
      k.p = points_3D[j].reshape((3, 1))

    return np.argwhere(converged == 1).reshape((-1,))


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