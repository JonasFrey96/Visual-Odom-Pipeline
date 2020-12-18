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
    self._extractor = Extractor()
    self._visu = Visualizer() # current frame, reproject the state (current landmarks), trajecktory plot from top XY
    self._H_latest = np.eye(4)
    self._state, self._t_loader, self._tra_gt = self._get_init_state()
    self._t_step = 1
    self._visu.plot_map(self._state, xlims=[-5, 20], zlims=[-5, 20])
    self._keyframes = [self._t_loader]

  def _transform_landmarks(self, T_rel):
    """Apply a transform to all 3D landmark positions"""
    for i, kp in enumerate(self._state._landmarks):
      p_new = T_rel @ np.concatenate([kp.p, np.ones((1, 1))], axis=0)
      self._state._landmarks[i].p = p_new[:3, :]

  def _update_landmark_desc(self, match_dict, kp, desc, t=None):
    for m in match_dict:
      self._state._landmarks[m.trainIdx].uv = np.array(kp[m.queryIdx].pt)
      # self._state._landmarks[m.trainIdx].desc = desc[m.queryIdx, :]
      if t:
        self._state._landmarks[m.trainIdx].t_latest = t

  def _select_keyframe(self, H_curr, angle_threshold=1.0, dist_threshold=1.0):
    """Keyframe selection based on angle and baseline"""
    t_prev = self._keyframes[-1]
    H_prev = self._state._trajectory._poses[t_prev]
    H_rel = H_curr @ np.linalg.inv(H_prev)

    rot_mat_rel = H_rel[:3, :3]
    rot_angle = np.abs(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rot_mat_rel).as_rotvec())))

    baseline = np.linalg.norm(H_rel[:3, 3])
    print(f"Baseline={baseline} // Angle={rot_angle}")
    return (rot_angle > angle_threshold) or (baseline > dist_threshold)

  def _get_init_state(self):
    t0, t1 = self._loader.getInit()
    print(f"[t0, t1]: {t0},{t1}")
    img0, H0_gt = self._loader.getFrame(t0)
    img1, H1_gt = self._loader.getFrame(t1)

    # feature extraction 
    kp_0, desc_0 = self._extractor.extract(img0)
    kp_1, desc_1 = self._extractor.extract(img1)
    # mat  # load first two frames 
    matches = self._extractor.match(kp_0, desc_0, kp_1, desc_1)

    # create the landmarks list
    landmarks = []
    landmarks_cor = []
    used_idx0 = []
    used_idx1 = []
    for m in matches:
      l = Keypoint(t0, t1, 2, uv=np.array(kp_0[m.queryIdx].pt), p=None, des=desc_0[m.queryIdx])
      l_cor = Keypoint(t0, t1, 2, uv=np.array(kp_1[m.trainIdx].pt), p=None, des=desc_1[m.trainIdx])
      used_idx0.append(m.queryIdx)
      used_idx1.append(m.trainIdx)
      landmarks.append(l)
      landmarks_cor.append(l_cor)
    K = self._loader.getCamera() 
    inliers, H1 = self._extractor.camera_pose(K, landmarks, landmarks_cor)
    landmarks = [landmarks[i] for i in inliers]
    landmarks_cor = [landmarks_cor[i] for i in inliers]
    print(f"Pipeline bootstrapped with {len(inliers)} correspondences")

    # Set baseline length = 1
    H1[:3, 3] /= np.linalg.norm(H1[:3, 3])

    # init the trajektory
    H0 = np.eye((4))
    converged = self._extractor.triangulate(K, H0, H1, landmarks, landmarks_cor)
    landmarks = [landmarks[i] for i in converged]
    landmarks_cor = [landmarks_cor[i] for i in converged]

    # Visualize landmarks
    self._visu.plot_landmarks(landmarks_cor, landmarks, img0, K)

    # create the candidate list 
    id0 = np.arange(0, len(kp_0) )
    id1 = np.arange(0, len(kp_1) )
    cand_id0 = np.delete(id0, np.array(used_idx0))
    cand_id1 = np.delete(id1, np.array(used_idx1))
    candidates = []
    # for i in range(cand_id0.shape[0]):
    #   c = Keypoint(t0, t0, 1, uv=kp_0[cand_id0[i]].pt, p=None, des=desc_0[cand_id0[i]])
    #   candidates.append(c)
    for i in range(cand_id1.shape[0]):
      c = Keypoint(t1, t1, 1, uv=np.array(kp_1[cand_id1[i]].pt), p=None, des=desc_1[cand_id1[i], :])
      candidates.append(c)

    # fill the two list with candidates and landmarks
    tra = Trajectory({t0: H0, t1: H1})
    tra_gt = Trajectory({t0: H0_gt, t1: H1_gt})
    self._H_latest = H1
    return State(landmarks_cor, candidates, tra), t1, tra_gt

  def step(self):
    self._t_loader += 1
    self._t_step += 1
    # load a new frame
    img0, H0_gt = self._loader.getFrame(self._t_loader)
    K = self._loader.getCamera()

    # Store true pose
    self._tra_gt._poses[self._t_loader] = H0_gt

    # TODO: KLT Tracking

    # [4.1] Associate keypoints to existing landmarks
    # feature extraction and matching
    kp_0, desc_0 = self._extractor.extract(img0)
    matches_land = self._extractor.match_list(kp_0, desc_0, self._state._landmarks)

    # Extract relevant landmarks for localization
    used_idx0 = []
    used_idx1 = []
    database_landmarks = []
    current_landmarks = []
    for m in matches_land:
      # Updated the landmarks_list
      used_idx0.append(m.queryIdx)
      used_idx1.append(m.trainIdx)
      self._state._landmarks[m.trainIdx].t_total += 1

      database_landmarks.append(deepcopy(self._state._landmarks[m.trainIdx]))
      current_landmarks.append(Keypoint(self._state._landmarks[m.trainIdx].t_first,
                                        self._t_loader,
                                        self._state._landmarks[m.trainIdx].t_total+1,
                                        np.array(kp_0[m.queryIdx].pt),
                                        None,
                                        desc_0[m.queryIdx, :]))

    # [4.2] Estimating current camera pose (relative to first image coordinate frame)
    self._visu.append_plot_num_landmarks(len(database_landmarks))
    inliers, H1 = self._extractor.camera_pose(K, database_landmarks, current_landmarks, corr='3D-2D')
    print(f"Num inliers = {len(inliers)}")

    database_landmarks = [database_landmarks[i] for i in inliers]
    current_landmarks = [current_landmarks[i] for i in inliers]
    matches_land = [matches_land[i] for i in inliers]

    # Visualize localization landmarks
    self._visu.plot_landmarks(database_landmarks, current_landmarks, img0, K,
                              T=H1)

    #  Reprojection error of the landmarks (that are present in current frame)
    err = self._extractor.reprojection_error(database_landmarks,
                                             current_landmarks, K, T=H1)
    self._visu.append_plot_reprojection_error(err)

    # Determine relative scale and re-scale translation
    rel_scale = self._extractor.relative_scale(K, self._state._trajectory, H1, database_landmarks,
                                               current_landmarks)
    H1[:3, 3] *= rel_scale/np.linalg.norm(H1[:3, 3])

    # Trajektory update (append new pose)
    self._state._trajectory.append(self._t_loader, H1)
    self._H_latest = H1

    # Check if frame should be used for triangulation
    if self._select_keyframe(H1):

      # Refine Landmarks maybe based on new obs(HOW TO UPDATE THE LANDMARKS CORRECTLY)
      self._update_landmark_desc(matches_land, kp_0, desc_0, t=self._t_loader)

      # TODO: Delete not used landmarks from landmark list

      # [4.3] Triangulate new landmarks using the previous keyframe
      used_idx0_c = []
      used_idx1_c = []
      removed_candidates = []
      # TODO: Should we only match against the latest keyframe?
      matches_cand = self._extractor.match_list(kp_0, desc_0,
                                                self._state._candidate)
      new_landmarks = []
      for m in matches_cand:
        # self._state._candidate[m.trainIdx].t_total += 1
        # self._state._candidate[m.trainIdx].t_latest = self._t_loader
        used_idx0_c.append(m.queryIdx)
        used_idx1_c.append(m.trainIdx)
        removed_candidates.append(m.trainIdx)

        kp_db = self._state._candidate[m.trainIdx]
        kp_new = Keypoint(kp_db.t_first, self._t_loader, kp_db.t_total,
                          np.array(kp_0[m.queryIdx].pt), None, desc_0[m.queryIdx, :])

        H0 = self._state._trajectory._poses[kp_db.t_latest]
        converged = self._extractor.triangulate(K, H0, H1, [kp_db], [kp_new])
        if len(converged) == 1:
          new_landmarks.append(kp_new)

      # Check new landmarks
      self._visu.plot_landmarks(new_landmarks, new_landmarks, img0, K, T=H1, window_name="Proposed Landmarks")
      self._state._landmarks += new_landmarks
      # cv2.waitKey(0)

      # Remove triangulated keypoints from candidates
      # TODO: Find a better way to do this
      updated_candidate = []
      for i, kp in enumerate(self._state._candidate):
        if i not in removed_candidates:
          updated_candidate.append(kp)
      self._state._candidate = updated_candidate
      print(f"New landmarks triangulated: {len(removed_candidates)}")

      # Add unused keypoints to candidates
      id = np.arange(0, len(kp_0))
      cand_id = np.delete(id, np.array(used_idx0_c))
      for i in range(cand_id.shape[0]):
        c = Keypoint(self._t_loader, self._t_loader, 1, uv=np.array(kp_0[cand_id[i]].pt), p=None,
                     des=desc_0[cand_id[i], :])
        self._state._candidate.append(c)

      # grouping into candidates and matched ones
      # triangulation

    #  2D map from the top
    self._visu.plot_map(self._state, self._tra_gt, xlims=[-20, 20], zlims=[-20, 20])

  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t_loader))
    
    for i, t in enumerate( range(self._t_loader, len(self._loader))):
      if i % 1 == 0:
        logging.info('Pipeline run ' + str( self._t_loader) +'/'+str( len(self._loader)))
       
      self.step()
