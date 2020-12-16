import numpy as np 
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def extract_features(img):
  # Initiate STAR detector
  orb = cv2.ORB_create()
  # find the keypoints with ORB
  kp = orb.detect(img,None)
  # compute the descriptors with ORB
  kp, des = orb.compute(img, kp)
  # draw only keypoints location,not size and orientation
  img2 = cv2.drawKeypoints(img,kp,img, color=(0,255,0), flags=0)
  return kp,des, img2


class Extractor():
  def __init__(self,cfg=None):
    self._cfg = cfg
    self._sift = cv2.SIFT_create(sigma=0.8)
    self._matcher = cv2.BFMatcher() 
    self._sift_ratio = 0.7
    
  def extract(self, img):

    kp_1, desc_1 = self._sift.detectAndCompute(img, None)
    return kp_1, desc_1

  def match(self, kp_1, desc_1, kp_2, desc_2):
    matches = self._matcher.knnMatch(desc_1, desc_2, k=2)
    good = []

    for m, n in matches:
        if m.distance < self._sift_ratio*n.distance:
          good.append(m)
    return good

  def match_list(self, kp_1, desc_1, keypoints):
    desc_2 = np.array([k.des for k in keypoints])
    kp_2 = [k.uv for k in keypoints]

    matches = self._matcher.knnMatch(desc_1, desc_2, k=2)
    good = []

    for m, n in matches:
        if m.distance < self._sift_ratio*n.distance:
          good.append(m)
    return good

  def camera_pose(self, K, land1, land2, corr='2D-2D'):
    if corr == '2D-2D':
      """Get pose from 2D-2D correspondences"""
      # Compute essential matrix
      kp_1_pts = np.array([kp.uv for kp in land1]).astype(np.float32)
      kp_2_pts = np.array([kp.uv for kp in land2]).astype(np.float32)
      E, mask = cv2.findEssentialMat(kp_1_pts, kp_2_pts, K, prob=0.999, method=cv2.RANSAC, mask=None, threshold=1)
      # Remove outliers and recover pose
      kp_1_pts = kp_1_pts[mask, :]
      kp_2_pts = kp_2_pts[mask, :]
      retval, R, t, mask = cv2.recoverPose(E, kp_1_pts, kp_2_pts, K)

    elif corr == '3D-2D':
      """Get pose from 3D-2D correspondences"""
      # Compute essential matrix
      kp_db_pts_3d = np.array([kp.p.T for kp in land1]).astype(np.float32)
      kp_curr_pts = np.array([kp.uv for kp in land2]).astype(np.float32)

      kp_db_pts_3d = kp_db_pts_3d.reshape((-1, 1, 3))
      kp_curr_pts = kp_curr_pts.reshape((-1, 1, 2))
      retval, rvec, t, inliers = cv2.solvePnPRansac(kp_db_pts_3d, kp_curr_pts,
                                                    K, None, reprojectionError=1.0)
      R, _ = cv2.Rodrigues(rvec)

    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = t.reshape((3,))
    return H
  
  def triangulate(self, K, H0, H1, keyp0, keyp1):
    """Triangulate a batch of keypoints between two images
     keypoints are passed by reference.
    Results will be stored in keyp0.p (Triangulated points are expressed
    in the coordinate frame of camera 0).

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
    points_4D = cv2.triangulatePoints(P_0, P_1, uv0, uv1).reshape((4, -1)).T
    points_3D = (points_4D/points_4D[:, 3].reshape((-1, 1)))[:, :3]

    for j, k in enumerate(keyp0):
      k.p = points_3D[j].reshape((3, 1))

  def reprojection_error(self, landmarks_3D, landmarks_2D, K, norm='L2', T=np.eye(4)):
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