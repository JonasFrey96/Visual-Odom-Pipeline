import numpy as np
from .trajectory import Trajectory

class State():
  def __init__( self, landmarks = [], candidates = [], trajectory=None):
    
    self._landmarks = landmarks
    self._candidate = candidates
    if not isinstance(trajectory, type(None)):
      self._trajectory = trajectory

  