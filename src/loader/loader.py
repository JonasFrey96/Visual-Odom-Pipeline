import numpy as np 
from PIL import Image
import cv2
class Loader():

  def __init__(self):
    self._name = 'Not defined'
    self._camera, self._poses, self.image_paths = self._loadData()
    self._length = 0 

  def __str__(self):
    return self._name

  def __len__(self):
    return self._length

  def _loadData(self):
    self._camera = None
    self._poses = []
    self._image_paths = []
    return self._camera, self._poses, self._image_paths

  def getImage(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError

    return cv2.imread(self._image_paths[id],0)  #Image.open( self._image_paths[id] )

  def getPose(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError

    return self._poses[id]

  def getFrame(self, id):
    if id >= self._length or id < 0: 
       raise AssertionError

    return getImage(id), getPose(id)
  
  def getCamera(self):
    return self._camera
