if __name__ == "__main__":  
  import sys
  sys.path.append('/home/jonas/Documents/Repos/Visual-Odom-Pipeline')

  
from src.camera import Camera
from src.loader import Loader
import numpy as np
import pandas as pd
from pathlib import Path

class ParkingLoader(Loader):

  def __init__(self, root):
    self._name = 'Parking Loader'
    self._root = root
    self._loadData()
    self._length = len(self._poses)

  def _loadData(self):
    K = pd.read_csv(f'{self._root}/K.txt', sep=",", header=None).to_numpy()[:3,:3]
    self._camera = Camera.from_K(K)
    p = pd.read_csv(f'{self._root}/poses.txt', sep=" ", header=None).to_numpy()[:,:-1]
    self._poses = [p[i,:] for i in range(p.shape[0])]
    self._image_paths = [str(pa) for pa in Path(self._root).rglob('*.png')]
    self._image_paths.sort()

def test():
  pl = ParkingLoader(root='/home/jonas/Documents/Datasets/parking')
  print(pl.getCamera())
  print(pl.getPose(0))
  print(pl.getImage(0))
  
  


if __name__ == "__main__":
  test()		