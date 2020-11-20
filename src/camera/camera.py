import numpy as np 

class Camera():

  def __init__(self, fx ,fy ,cx ,cy):
    self._K = np.eye((3))
    self._K[0,0] = fx
    self._K[1,1] = fy
    self._K[2,0] = cx
    self._K[2,1] = cy

  def __str__(self):
    return 'Camera with K: ' + str(self._K)
    
  @classmethod
  def from_K(self,K):
    return Camera(fx=K[0,0], fy=K[1,1],cx=K[2,0],cy=K[2,1])