import numpy as np

class Trajectory():
  def __init__( self, poses= []):
    self._poses = poses

  def __len__(self):
    return len(self._poses)

  def __str__(self):
    s = '='*60
    s += f'\nTrajektory Length {len(self)} \n'
    if len(self)> 2:
      s += f'Start:  \n{self._poses[0]} \n'
      s += f'Stop:  \n{self._poses[-1]} \n'
    s += '='*60
    return s
    
  def append(self,pose):
    self._poses.append(pose)
  
  def remove(self, i):
    if i > len(self):
      raise ValueError('Out of bounds')
    self._poses = self._poses[:i] + self._poses[i:]

  def relative(self, i,j):
    pass

if __name__ == "__main__":
  from scipy.stats import special_ortho_group
  tra = Trajectory()
  for i in range(0,10):
    h = np.eye(4)
    h[:3,:3] = special_ortho_group.rvs(3)
    tra.append( h )
  
  print(tra)
      
