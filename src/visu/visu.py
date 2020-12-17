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
    self._localized_landmarks = []
    self._reprojection_errors = []

  def within_image(self, uv, img_dims):
    """Given xy pixel coordinate and the img shape (width, height),
    return boolean indicating whether the pixel lies inside the image bounds"""
    return 0 <= uv[0] <= img_dims[0] and 0 <= uv[1] <= img_dims[1]

  def append_plot_reprojection_error(self, err_reproj):
    """Add an entry to the history of num localized landmarks.
    Then plot the entire history."""
    self._reprojection_errors.append(err_reproj)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(range(len(self._reprojection_errors)), self._reprojection_errors)
    plt.title("Landmark Reprojection Error")
    plt.xlabel("Frame Index")
    plt.ylabel("Reprojection Error")
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow("Landmark Reprojection Error", data)
    cv2.waitKey(1)

  def append_plot_num_landmarks(self, num_landmarks):
    """Add an entry to the history of num localized landmarks.
    Then plot the entire history."""
    self._localized_landmarks.append(num_landmarks)

    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(range(len(self._localized_landmarks)), self._localized_landmarks)
    plt.title("Number of Landmarks used for Localization")
    plt.xlabel("Frame Index")
    plt.ylabel("Number of Landmarks")
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow("Num Landmarks", data)
    cv2.waitKey(1)

  def plot_map(self, state, tra_gt=None, xlims=None, zlims=None):
    """Plot the 3D landmarks and the trajectory"""
    fig = plt.figure()
    fig.add_subplot(111)
    landmark_pts_3D = np.array([l.p for l in state._landmarks]).reshape((-1, 3))
    plt.plot(landmark_pts_3D[:, 0], landmark_pts_3D[:, 2], marker='o', linestyle='', color='green')

    # Extract estimated trajectory
    tra_pos = []
    for t in state._trajectory._poses.keys():
      H = state._trajectory._poses[t]
      R, t = H[:3, :3], H[:3, 3].reshape((3, 1))
      tra_pos.append(-(R.T @ t).T)
    tra_pos = np.array(tra_pos).reshape((-1, 3))


    # Extract gt trajectory
    if tra_gt:
      tra_pos_gt = []
      for t in state._trajectory._poses.keys():
        H = tra_gt._poses[t]
        R, t = H[:3, :3], H[:3, 3].reshape((3, 1))
        tra_pos_gt.append((R.T @ t).T)
      tra_pos_gt = np.array(tra_pos_gt).reshape((-1, 3))

      # Scale correction
      baseline = np.linalg.norm(tra_pos, axis=1)
      baseline_gt = np.linalg.norm(tra_pos_gt, axis=1)
      scale = np.mean(np.divide(baseline, baseline_gt))
      tra_pos_gt *= scale

      plt.plot(tra_pos_gt[:, 0], tra_pos_gt[:, 2], color='red', linestyle='dashed')
      plt.plot(tra_pos_gt[-1, 0], tra_pos_gt[-1, 2], color='red', marker='o')

    # Plot trajectories
    plt.plot(tra_pos[:, 0], tra_pos[:, 2], color='blue', linestyle='dashed')
    plt.plot(tra_pos[-1, 0], tra_pos[-1, 2], color='blue', marker='o')

    if xlims:
      plt.xlim(xlims)
    if zlims:
      plt.ylim(zlims)

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow("Map", data)
    cv2.waitKey(1)

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
    cv2.waitKey(1)

  # def plot_img(self, img, tag='img', store=True):
  #   pil_img = Image.fromarray( np.uint8(img) ,'RGB')
  #   pil_img.save(self.p + '/' + tag + '.png')
