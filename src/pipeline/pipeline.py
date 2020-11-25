import numpy as np
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

  def _get_init_state(self):
    t0, t1 = self._loader.getInit()
    print(t0,t1)
    img0, H0_gt = self._loader.getFrame(t0)
    img1, H1_gt = self._loader.getFrame(t1)

    # feature extraction 
    kp_0,desc_0 = self._extractor.extract(img0)
    kp_1,desc_1 = self._extractor.extract(img1)
    # mat  # load first two frames 
    matches = self._extractor.match(kp_0,desc_0,kp_1,desc_1)

    # create the landmarks list
    landmarks = []
    landmarks_cor = []
    used_idx0 = []
    used_idx1 = []
    for m in matches:
      # TODO calculate the p=NONE
      l = Keypoint(0,0,2, uv = kp_0[m.queryIdx].pt, p=None, des = desc_0[m.queryIdx])
      l_cor = Keypoint(0,0,2, uv = kp_1[m.trainIdx].pt, p=None, des = desc_1[m.trainIdx])
      used_idx0.append( m.queryIdx )
      used_idx1.append( m.trainIdx )
      landmarks.append(l)
      landmarks_cor.append(l_cor)
    K = self._loader.getCamera() 
    H1 = self._extractor.camera_pose(K, landmarks, landmarks_cor)

    # init the trajektory
    H0 = np.eye((4))
    self._extractor.triangulate(K,H0, H1, landmarks, landmarks_cor)

    # create the candidate list 
    id0 = np.arange(0, len(kp_0) )
    id1 = np.arange(0, len(kp_1) )
    cand_id0 = np.delete(id0, np.array( used_idx0) )
    cand_id1 = np.delete(id1, np.array( used_idx1) )
    candidates = []
    for i in range(cand_id0.shape[0]):
      c = Keypoint(0,0,1, uv = kp_0[cand_id0[i]].pt, p=None, des = desc_0[cand_id0[i]])
      candidates.append(c)
    for i in range(cand_id1.shape[0]):
      c = Keypoint(0,0,1, uv = kp_1[cand_id1[i]].pt, p=None, des = desc_1[cand_id1[i]])
      candidates.append(c)

    # fill the two list with candidates and landmarks (5 point algo) 
    
    tra = Trajectory([H0,H1])

    return State(landmarks, candidates, tra), t1+1

  def step(self):
    self._t_loader += 1
    self._t_step += 1
    img0, H0_gt = self._loader.getFrame(self._t_loader)
    # load a new frame 
    # feature extraction
    # feature extraction 
    kp_0,desc_0 = self._extractor.extract(img0)
    # mat  # load first two frames 
    
    matches_land = self._extractor.match_list(kp_0,desc_0, self._state._landmarks)
    # create the landmarks list
    
    used_idx0 = []
    used_idx1 = []
    for m in matches_land:
      # Updated the landmarks_list
      self._state._landmarks[ m.trainIdx ].t_total += 1
      self._state._landmarks[ m.trainIdx ].t_latest = self._t_step
      used_idx0.append( m.queryIdx )
      used_idx1.append( m.trainIdx )

    # get new camera pose 
    K = self._loader.getCamera() 
    H1 = self._extractor.camera_pose(K, landmarks, landmarks_cor)



    used_idx0_c = []
    used_idx1_c = []
    matches_cand = self._extractor.match_list(kp_0,desc_0, self._state._candidate)
    for m in matches_cand:
      self._state._candidate[ m.trainIdx ].t_total += 1
      self._state._candidate[ m.trainIdx ].t_latest = self._t_step
      used_idx0_c.append( m.queryIdx )
      used_idx1_c.append( m.trainIdx )

      if self._state._candidate[ m.trainIdx ].t_total > 3:
        self._state._landmarks.append( self._state._candidate.pop( m.trainIdx ) )
        # triangulate new keypoints (for all ?)


    # grouping into candidates and matched ones
    # triangulation 
    # trajektory update (append new pose)

    # check if the frame should be used. min trajektory distance and angle

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
    #  Current landmarks projected onto the image for Two Frames with configurable delta T!
    #  reprojection error of the landmarks
    

  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t_loader))
    
    for i, t in enumerate( range(self._t_loader, len(self._loader))):
      if i % 10 == 0:
        logging.info('Pipeline run ' + str( self._t_loader) +'/'+str( len(self._loader)))
       
      self.step()

  
