import numpy as np
from .trajectory import Trajectory

class State():
  def __init__( self, landmarks = [], candidates = [], trajectory= ):
    
    self._landmarks = landmarks
    self._candidate = candidates
    if self trajectory == None:
      trajectory = Trajectory()
  
  def from_init_res(res)
    # process res to init lists 
    return State('a','b','c')