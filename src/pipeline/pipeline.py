import numpy as np
from state import State
from visu import Visualizer
import logging

class Pipeline():
  def __init__(self, loader):
    self._t = 0
    self._state = self._get_init_state() 
    self._visu = Visualizer() # current frame, reproject the state (current landmarks), trajecktory plot from top XY
    self._loader = loader

  def _get_init_state(self):
    # load first two frames 
    # feature extraction and matching
    # fill the two list with candidates and landmarks (5 point algo) 
    # init the trajektory
    return State()

  def step(self):
    # load a new frame 
    # feature extraction
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
    
    self._t += 1

  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t))
    
    for i, t in enumerate( range(self._t, len(self._loader))):
      if i % 10 == 0:
        logging.info('Pipeline run ' + str( self._t) +'/'+str( len(self._loader)))
       
      self.step()

  
