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
    # here goes the pipeline initalization code 
    return State()

  def step(self):
    # feature extraction what so ever udating the state 
    self._t += 1

  def full_run(self):
    logging.info('Started Full run at timestep '+ str(self._t))
    
    for i, t in enumerate( range(self._t, len(self._loader))):
      if i % 10 == 0:
        logging.info('Pipeline run ' + str( self._t) +'/'+str( len(self._loader)))
       
      self.step()
