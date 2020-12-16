from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation
from state import State
from visu import Visualizer
import logging
from extractor import Extractor
from state import Keypoint, Trajectory

class Pipeline():
  def __init__(self, loader):
    self._loader = loader
    self._extractor = Extractor()
    self._visu = Visualizer() # current frame, reproject the state (current landmarks), trajecktory plot from top XY
    self._state, self._t_loader = self._get_init_state() 
    self._t_step = 1

  def _transform_landmarks(self, T_rel):
    """Apply a transform to all 3D landmark positions"""
    for i, kp in enumerate(self._state._landmarks):
      p_new = T_rel @ np.concatenate([kp.p, np.ones((1, 1))], axis=0)
      self._state._landmarks[i].p = p_new[:3, :]

  def _select_keyframe(self, H_curr, angle_threshold=0.01):
    """Keyframe selection based on angle"""
    # TODO: keyframe selection based on relative pose instead of absolute
    t_latest = max(self._state._trajectory._poses.keys())
    H_latest = self._state._trajectory._poses[t_latest]
    rot_mat_latest = H_latest[:3, :3]
    rot_mat = H_curr[:3, :3]
    rot_mat_rel = rot_mat_latest @ rot_mat.T
    rot_angle = np.abs(np.rad2deg(np.linalg.norm(Rotation.from_matrix(rot_mat_rel).as_rotvec())))
    print(f"Rotation Angle: {rot_angle}")
    return rot_angle > angle_threshold

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
      # TODO calculate the p=NONE
      l = Keypoint(0, 0, 2, uv=np.array(kp_0[m.queryIdx].pt), p=None, des=desc_0[m.queryIdx])
      l_cor = Keypoint(0, 0, 2, uv=np.array(kp_1[m.trainIdx].pt), p=None, des=desc_1[m.trainIdx])
      used_idx0.append(m.queryIdx)
      used_idx1.append(m.trainIdx)
      landmarks.append(l)
      landmarks_cor.append(l_cor)
    K = self._loader.getCamera() 
    H1 = self._extractor.camera_pose(K, landmarks, landmarks_cor)

    # init the trajektory
    H0 = np.eye((4))
    self._extractor.triangulate(K, H0, H1, landmarks, landmarks_cor)

    # Visualize landmarks
    self._visu.plot_landmarks(landmarks, landmarks, img0, K)

    # create the candidate list 
    id0 = np.arange(0, len(kp_0) )
    id1 = np.arange(0, len(kp_1) )
    cand_id0 = np.delete(id0, np.array(used_idx0))
    cand_id1 = np.delete(id1, np.array(used_idx1))
    candidates = []
    for i in range(cand_id0.shape[0]):
      c = Keypoint(0, 0, 1, uv=kp_0[cand_id0[i]].pt, p=None, des=desc_0[cand_id0[i]])
      candidates.append(c)
    for i in range(cand_id1.shape[0]):
      c = Keypoint(0, 0, 1, uv=kp_1[cand_id1[i]].pt, p=None, des=desc_1[cand_id1[i]])
      candidates.append(c)

    # fill the two list with candidates and landmarks (5 point algo)
    tra = Trajectory({t0: H0, t1: H1})

    return State(landmarks, candidates, tra), t1+1

  def step(self):
    self._t_loader += 1
    self._t_step += 1
    # load a new frame
    img0, H0_gt = self._loader.getFrame(self._t_loader)
    K = self._loader.getCamera()

    # TODO: KLT Tracking

    # [4.1] Associate keypoints to existing landmarks
    # feature extraction and matching
    kp_0, desc_0 = self._extractor.extract(img0)
    matches_land = self._extractor.match_list(kp_0, desc_0, self._state._landmarks)

    # create the landmarks list
    used_idx0 = []
    used_idx1 = []
    database_landmarks = []
    current_landmarks = []
    for m in matches_land:
      # Updated the landmarks_list
      self._state._landmarks[m.trainIdx].t_total += 1
      self._state._landmarks[m.trainIdx].t_latest = self._t_loader
      used_idx0.append(m.queryIdx)
      used_idx1.append(m.trainIdx)

      database_landmarks.append(self._state._landmarks[m.trainIdx])
      current_landmarks.append(Keypoint(self._state._landmarks[m.trainIdx].t_first,
                                        self._t_loader,
                                        self._state._landmarks[m.trainIdx].t_total+1,
                                        np.array(kp_0[m.queryIdx].pt),
                                        None,
                                        desc_0[m.queryIdx, :]))

    # Visualization check (projected landmarks should be close)
    t_latest = max(self._state._trajectory._poses.keys())
    H_latest = self._state._trajectory._poses[t_latest]
    self._visu.plot_landmarks(database_landmarks, current_landmarks, img0, K, T=H_latest)

    # [4.2] Estimating current camera pose (relative to first image coordinate frame)
    H1 = self._extractor.camera_pose(K, database_landmarks, current_landmarks, corr='3D-2D')

    # [4.3] Triangulate new landmarks
    used_idx0_c = []
    used_idx1_c = []
    matches_cand = self._extractor.match_list(kp_0, desc_0, self._state._candidate)
    for m in matches_cand:
      self._state._candidate[m.trainIdx].t_total += 1
      self._state._candidate[m.trainIdx].t_latest = self._t_loader
      used_idx0_c.append(m.queryIdx)
      used_idx1_c.append(m.trainIdx)

      if self._state._candidate[m.trainIdx].t_total > 3:
        kp_db = deepcopy(self._state._candidate[m.trainIdx])
        # triangulate new keypoints (for all ?)
        kp_new = deepcopy(self._state._landmarks[-1])
        kp_new.uv = np.array(kp_0[m.queryIdx].pt)
        kp_new.des = desc_0[m.queryIdx, :]
        H0 = self._state._trajectory._poses[kp_db.t_first]
        self._extractor.triangulate(K, H0, H1, [kp_db], [kp_new])

        # Bring new landmark into coordinate frame of first image
        # TODO: Is it wrong for the uv and desc be updated? Not sure if it matters
        kp_new.p = (H0 @ np.concatenate([kp_new.p, np.ones((1, 1))], axis=0))[:3, :]
        self._state._landmarks.append(kp_new)

    # TODO: Find a better way to do this
    updated_candidate = []
    removal_idxs = [m.trainIdx for m in matches_cand]
    for i, kp in enumerate(self._state._candidate):
      if i not in removal_idxs:
        updated_candidate.append(kp)
    self._state._candidate = updated_candidate

    # grouping into candidates and matched ones
    # triangulation

    # Trajektory update (append new pose)
    # check if the frame should be used. min trajektory distance and angle
    if self._select_keyframe(H1):
      self._state._trajectory.append(self._t_loader, H1)

      # if we add it:
      # update candidates list
      # update the landmark list

      # Refine Landmarks maybe based on new obs(HOW TO UPDATE THE LANDMARKS CORRECTLY)

      # Move candidates to landmark list
      # Delete unused candidates
      # Delete not used landmarks from landmark list

      # Visu and state plotting
      #  Overview of the lists

      #  2D map from the top
      self._visu.plot_map(self._state)

      #  Current landmarks projected onto the image for Two Frames with configurable delta T!
      self._visu.plot_landmarks(database_landmarks, current_landmarks, img0, K, T=H1)

      #  reprojection error of the landmarks (that are present in current frame)
      err_reproj = self._extractor.reprojection_error(database_landmarks, current_landmarks, K, T=H1)
      print(f"Current Frame Mean Re-projection Error: {err_reproj}")

  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t_loader))
    
    for i, t in enumerate( range(self._t_loader, len(self._loader))):
      if i % 1 == 0:
        logging.info('Pipeline run ' + str( self._t_loader) +'/'+str( len(self._loader)))
       
      self.step()
