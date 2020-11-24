import os
from os.path import expanduser
from pathlib import Path

class Visualizer():
  def __init__(self, path= None):
    if path is None:
      path = expanduser("~")+'/visu'
      Path(path).mkdir(parents=True, exist_ok=True)
    self._p = path

  def plot_landmarks(self, landmarks, img, K):
    pass

  def plot_img(self, img, tag='img', store=True):
    pil_img = Image.fromarray( np.uint8(img) ,'RGB')
    pil_img.save(self.p + '/' + tag + '.png')
