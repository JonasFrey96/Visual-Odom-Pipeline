import numpy as np 
from PIL import Image
import cv2
from pathlib import Path
import os
import argparse
import yaml
import pandas as pd


class Loader():

  def __init__(self, name, cfg):
    self._name = name
    self._cfg = cfg
    self._bilateral_filter_params = {
      'd': 5,
      'sigmaColor': 1.5,
      'sigmaSpace': 1.5
    }
    self._camera, self._poses, self.image_paths = self._loadData()
    self._length = self._poses.shape[0] 
    
  def __str__(self):
    return self._name

  def __len__(self):
    return self._length

  def _loadData(self):
    cfg = self._cfg[self._name]
    base = cfg['path']
    if self._name == 'parking':
      print(base, 'WORKING')
      p = base + '/images'
      self._image_paths = [str(p) for p in Path(p).rglob('*.png')]
      ar = np.reshape( np.loadtxt(base+'/poses.txt' ), (-1,3,4))
      self._poses = np.zeros( (ar.shape[0],4,4) )
      self._poses[:,3,3] = 1
      self._poses[:,:3,:] = ar
      data = pd.read_csv(base+'/K.txt', header = None) 

      self._camera = data.to_numpy()[:3, :3]
    elif self._name in ['roomtour', 'stairway', 'outdoor_street', 'outdoor_loop']:
      print(base, 'WORKING')
      p = base + '/images'
      self._image_paths = [str(p) for p in Path(p).rglob('*.png')]      
      sub = 1
      self._image_paths = self._image_paths[::sub]
      self._poses = np.zeros( (len(self._image_paths),4,4) )
      self._poses[:,3,3] = 1
      data = pd.read_csv(base+'/K.txt', header = None) 
      self._camera = data.to_numpy()[:3, :3]
      

    elif self._name == 'malaga':
      p = base + '/malaga-urban-dataset-extract-07_rectified_1024x768_Images'
      self._image_paths = [str(p) for p in Path(p).rglob('*_left.jpg')]
      self._poses = np.zeros((len(self._image_paths), 4, 4))
      self._camera = np.eye(3)
      with open(base + '/camera_params_rectified_a=0_1024x768.txt') as param_f:
        lines = param_f.readlines()
        self._camera[0, 0] = lines[8][3:-1]
        self._camera[0, 2] = lines[6][3:-1]
        self._camera[1, 1] = lines[9][3:-1]
        self._camera[1, 2] = lines[7][3:-1]

    elif self._name == 'kitti':
      p = base + '/00/image_0'
      self._image_paths = [str(p) for p in Path(p).rglob('*.png')]
      self._camera = np.genfromtxt(base + '/00/calib.txt')[0, 1:].reshape((3, 4))[:, :3]

      ar = np.reshape( np.loadtxt(base+'/poses/00.txt' ), (-1,3,4))
      self._poses = np.zeros( (ar.shape[0],4,4) )
      self._poses[:,3,3] = 1
      self._poses[:,:3,:] = ar
    else:
      raise Exception
    self._image_paths.sort()
    
    return self._camera, self._poses, self._image_paths

  def getImage(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError
    return cv2.bilateralFilter(cv2.imread(self._image_paths[id] , cv2.IMREAD_GRAYSCALE), **self._bilateral_filter_params) #Image.open( self._image_paths[id] )
    return cv2.imread(self._image_paths[id] , cv2.IMREAD_GRAYSCALE)

  def getPose(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError

    return self._poses[id]

  def getFrame(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError

    return self.getImage(id), self.getPose(id)
  
  def getCamera(self):
    return self._camera

  def getInit(self):
    """
    Returns Tuple with first and second index
    """
    return tuple(self._cfg[self._name]['init'])


if __name__ == "__main__":
  with open('/home/jonfrey/Visual-Odom-Pipeline/src/loader/datasets.yml') as f:
    doc = yaml.load(f, Loader=yaml.FullLoader)
  print(doc)

  loader = Loader('parking', doc)
  for i in range(len(loader)):
    img ,pose = loader.getFrame(i)
    print(pose, img.shape)