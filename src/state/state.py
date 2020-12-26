import numpy as np
from .trajectory import Trajectory

class State():
  def __init__( self, landmarks, tracked_kp, trajectory):
    
    self._landmarks = landmarks
    self._tracked_kp = tracked_kp
    self._trajectory = trajectory
