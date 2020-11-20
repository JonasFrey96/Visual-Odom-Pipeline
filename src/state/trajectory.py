import numpy as np

class Trajectory():
  def __init__( self, poses= []):
    self._poses = poses

  def __len__(self):
    return len(self._poses)

  def append(self,pose):
    self._poses.append(pose)
  
  def remove(self, i):
    if i > len(self):
      raise ValueError('Out of bounds')
    self._poses = self._poses[:i] + self._poses[i:]

  def relative(self, i,j):
    pass
  