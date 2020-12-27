import numpy as np
from .trajectory import Trajectory

class State():
  def __init__( self, landmarks, landmarks_kp, candidates_kp, trajectory):
    
    self._landmarks = landmarks
    self._landmarks_kp = landmarks_kp
    self._candidates_kp = candidates_kp
    self._trajectory = trajectory
