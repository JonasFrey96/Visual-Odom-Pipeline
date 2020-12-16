import os
from os.path import expanduser
from pathlib import Path

from random import randint
import cv2
import numpy as np


class Visualizer():
  def __init__(self, path= None):
    if path is None:
      path = expanduser("~")+'/visu'
      Path(path).mkdir(parents=True, exist_ok=True)
    self._p = path

  def plot_landmarks(self, landmarks, img, K):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for l in landmarks:
      uv_homo = K @ l.p.T
      uv = (uv_homo[:2]/uv_homo[2]).astype(np.int)
      uv_kpt = l.uv.astype(np.int)

      # TODO: Check image bounds before drawing
      color = tuple([randint(0, 256) for x in range(3)])
      cv2.circle(img, (uv_kpt[0], uv_kpt[1]), 2, color, -1)
      cv2.circle(img, (uv[0], uv[1]), 2, color, -1)
      cv2.line(img, (uv_kpt[0], uv_kpt[1]), (uv[0], uv[1]), color, 2)

    cv2.imshow("Projected Landmarks", img)
    cv2.waitKey(0)

  # def plot_img(self, img, tag='img', store=True):
  #   pil_img = Image.fromarray( np.uint8(img) ,'RGB')
  #   pil_img.save(self.p + '/' + tag + '.png')
