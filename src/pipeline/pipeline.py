from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from state import State
from visu import Visualizer
import logging
from extractor import Extractor
from state import Keypoint, Trajectory
import cv2

class Pipeline():
  def __init__(self, loader):
    self._loader = loader
    self._K = loader.getCamera()
    self._extractor = Extractor()
    self._visu = Visualizer(self._K)
    self._t_step = 1
    self._state, self._t_loader, self._tra_gt = self._get_init_state()
    self._extractor._im_prev = self._loader.getImage(self._t_loader)
    self._keyframes = [0, 1]

    self._visu.update(self._loader.getImage(self._t_loader), self._state)
    self._visu.render()

  def _select_keyframe(self, H):
    return False

  def _get_init_state(self):
    # Feature detection and matching
    t0, t1 = self._loader.getInit()
    im0, H0_gt = self._loader.getFrame(t0)
    im1, H1_gt = self._loader.getFrame(t1)

    # Initialize with SIFT kp's and descriptors
    kp0 = self._extractor.extract(im0, 0, detector='sift')
    kp1 = self._extractor.extract(im1, 1, detector='sift')
    matches = self._extractor.match_lists(kp0, kp1)

    # Split keypoints into matched (kp1_m, kp2_m) and un-matched (kp1_nm, kp2_nm)
    kp0_m, kp1_m = [], []
    i0_nm, i1_nm = list(range(len(kp0))), list(range(len(kp1)))
    for match in matches:
      kp0_m.append(kp0[match.queryIdx])
      kp1_m.append(kp1[match.trainIdx])
      if match.queryIdx in i0_nm:
        i0_nm.remove(match.queryIdx)
      if match.trainIdx in i1_nm:
        i1_nm.remove(match.trainIdx)

    kp0_nm = [kp0[i] for i in i0_nm]
    kp1_nm = [kp1[i] for i in i1_nm]

    # Estimate homography (bootstrapped baseline length set = 1)
    H0 = np.eye(4)
    inliers, H1 = self._extractor.camera_pose(self._K, kp0_m, kp1_m, corr='2D-2D')

    # Remove outliers
    kp0_m = [kp0_m[i] for i in inliers]
    kp1_m = [kp1_m[i] for i in inliers]

    # Triangulate inliers to create landmarks
    converged, landmarks = self._extractor.triangulate(self._K, H0, H1, kp0_m, kp1_m, self._t_step)

    # Remove non-converged landmarks
    # kp0_m = [kp0_m[i] for i in converged]
    kp1_m = [kp1_m[i] for i in converged]

    # Detect new keypoints for tracking
    tracked_kp = kp1_m + kp1_nm

    # Initialize estimated and gt trajectories
    trajectory, trajectory_gt = Trajectory(), Trajectory()
    trajectory.append(0, H0)
    trajectory.append(1, H1)
    trajectory_gt.append(0, H0_gt)
    trajectory_gt.append(1, H1_gt)
    return State(landmarks, tracked_kp, trajectory), t1, trajectory_gt

  def step(self):
    self._t_step += 1
    self._t_loader += 1
    im, H_gt = self._loader.getFrame(self._t_loader)

    # Extend track lengths (Remove points that failed to track into current frame)
    self._state._tracked_kp = self._extractor.extend_tracks(im, self._state._tracked_kp)

    # Obtain 3D-2D Correspondences
    matches = self._extractor.match_lists(self._state._landmarks, self._state._tracked_kp)

    landmarks_m, tracked_kp_m = [], []
    for match in matches:
      landmarks_m.append(self._state._landmarks[match.queryIdx])
      tracked_kp_m.append(self._state._tracked_kp[match.trainIdx])

    # Localize with tracked keypoints
    inliers, H1 = self._extractor.camera_pose(self._K, landmarks_m, tracked_kp_m, corr='3D-2D')

    # Remove unused landmarks (un-matched, outliers)
    self._state._landmarks = [landmarks_m[i] for i in inliers]

    # Update trajectory
    self._state._trajectory.append(self._t_step, H1)

    # Triangulate if keyframe
    if self._select_keyframe(H1):
      landmarks_new, self._state._tracked_kp = self._extractor.triangulate_tracks(self._state._tracked_kp, self._state._trajectory,
                                                                                  min_track_length=5)
      self._state._landmarks += landmarks_new

    # Detect new features and initialize new tracks
    kp_new = self._extractor.extract(im, self._t_step, self._state._tracked_kp, detector='sift')
    self._state._tracked_kp += kp_new

    # Update visualizer
    self._visu.update(im, self._state)
    self._visu.render()


  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t_loader))
    
    for i, t in enumerate( range(self._t_loader, len(self._loader))):
      if i % 1 == 0:
        logging.info('Pipeline run ' + str( self._t_loader) +'/'+str( len(self._loader)))
       
      self.step()
