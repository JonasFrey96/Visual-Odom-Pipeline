import numpy as np
from .trajectory import Trajectory

class State():
  def __init__( self, landmarks = [], candidates = [], trajectory= None):
    
    self._landmarks = landmarks
    self._candidate = candidates
    if trajectory == None:
      self._trajectory = Trajectory()
  