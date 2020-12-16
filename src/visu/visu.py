import os
from os.path import expanduser
from pathlib import Path

from copy import deepcopy
from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer():
  def __init__(self, path= None):
    if path is None:
      path = expanduser("~")+'/visu'
      Path(path).mkdir(parents=True, exist_ok=True)
    self._p = path

  def within_image(self, uv, img_dims):
    """Given xy pixel coordinate and the img shape (width, height),
    return boolean indicating whether the pixel lies inside the image bounds"""
    return 0 <= uv[0] <= img_dims[0] and 0 <= uv[1] <= img_dims[1]

  def plot_map(self, state, xlims=None, zlims=None):
    """Plot the 3D landmarks and the trajectory"""
    plt.figure()
    landmark_pts_3D = np.array([l.p for l in state._landmarks]).reshape((-1, 3))
    plt.scatter(landmark_pts_3D[:, 0], landmark_pts_3D[:, 2])

    # Construct trajectory vector
    traj_pos = []
    for t in state._trajectory._poses.keys():
      traj_pos.append(state._trajectory._poses[t][:3, 3])
    traj_pos = np.array(traj_pos).reshape((-1, 3))
    plt.plot(traj_pos[:, 0], traj_pos[:, 2], 'r-')

    if xlims:
      plt.xlim(xlims)
    if zlims:
      plt.ylim(zlims)
    plt.show()

  def plot_landmarks(self, landmarks_3D, landmarks_2D, img, K, T=np.eye(4)):
    """Plot the projected 3D landmarks, connect to the corresponding
    2D landmarks with a line of the same color.

    Optionally transform 3D points with the homogeneous transform T"""
    img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for i in range(len(landmarks_2D)):
      l_2D, l_3D = deepcopy(landmarks_2D[i]), deepcopy(landmarks_3D[i])
      l_3D.p = (T @ np.concatenate([l_3D.p.reshape((3, 1)), np.ones((1, 1))]))[:3, :]
      uv_homo = K @ l_3D.p
      uv = (uv_homo[:2]/uv_homo[2]).astype(np.int)
      uv_kpt = l_2D.uv.astype(np.int)

      # TODO: Check image bounds before drawing
      if self.within_image(uv, img.shape[:2][::-1]) and self.within_image(uv_kpt, img.shape[:2][::-1]):
        color = tuple([randint(0, 256) for x in range(3)])
        cv2.circle(img, (uv_kpt[0], uv_kpt[1]), 2, color, -1)
        cv2.circle(img, (uv[0], uv[1]), 2, color, -1)
        cv2.line(img, (uv_kpt[0], uv_kpt[1]), (uv[0], uv[1]), color, 2)

    cv2.imshow("Projected Landmarks", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Projected Landmarks")

  # def plot_img(self, img, tag='img', store=True):
  #   pil_img = Image.fromarray( np.uint8(img) ,'RGB')
  #   pil_img.save(self.p + '/' + tag + '.png')
