import os
from os.path import expanduser
from pathlib import Path

from copy import deepcopy
from random import randint
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer():
  """
  Visualize state of the odometry pipeline.
  -Trajectory
  -Landmarks
  -Tracked Keypoints
  """
  def __init__(self, K, path=None):
    if path is None:
      path = expanduser("~")+'/visu'
      Path(path).mkdir(parents=True, exist_ok=True)
    self._p = path
    self._im = None
    self._landmarks = []
    self._landmark_history = []
    self._position_history = []
    self._position_gt_history = []
    self._tracked_px = []
    self._H_latest = np.eye(4)
    self._K = K

    self._fig = plt.figure(figsize=(12, 6))

  def within_image(self, uv, img_dims):
    """Given xy pixel coordinate and the img shape (width, height),
    return boolean indicating whether the pixel lies inside the image bounds"""
    return 0 <= uv[0] <= img_dims[0] and 0 <= uv[1] <= img_dims[1]

  def update(self, im, state):
    """
    Update the state of the visualizer. K and pose are used to project
    the landmarks into the current image. Convert both landmark list and kp
    list to a list of pixel coordinates
    """
    # Update image
    self._im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Store landmarks
    self._landmarks = state._landmarks
    self._landmark_history.append(len(self._landmarks))

    # Store tracked keypoints
    self._tracked_px = np.asarray([kp.uv for kp in state._tracked_kp]).reshape((-1, 2))

    # Store trajectory
    H = state._trajectory[len(state._trajectory)-1]
    position = H[:3, :3].T @ (-H[:3, 3].reshape((-1, 1)))
    self._position_history.append(position)

    # Store pose for projecting landmarks
    self._H_latest = H


  def render(self):
    self._fig = plt.figure(figsize=(12, 6))
    ax = self._fig.add_subplot(221)
    plt.title("Landmarks and Keypoints")
    ax.imshow(self._im)

    # Project landmarks
    landmarks_3d = np.array([l.p for l in self._landmarks]).reshape((-1, 3))
    landmarks_3d = np.concatenate([landmarks_3d, np.ones((len(landmarks_3d), 1))], axis=1)
    landmarks_3d = ((self._H_latest @ landmarks_3d.T).T)[:, :3]
    landmarks_px = (self._K @ landmarks_3d.T).T
    landmarks_px = (landmarks_px / landmarks_px[:, 2:3])[:, :2]
    ax.scatter(landmarks_px[:, 0], landmarks_px[:, 1], s=2, c='green', facecolor=None)

    # Draw tracked keypoints
    ax.scatter(self._tracked_px[:, 0], self._tracked_px[:, 1], s=2, c='red', facecolor=None)
    ax.legend(["Landmarks", "Tracked"])

    # Draw trajectory
    ax = self._fig.add_subplot(245)
    plt.title("# Landmarks")
    ax.plot(list(range(len(self._landmark_history))), self._landmark_history)

    ax = self._fig.add_subplot(246)
    plt.title("Global Trajectory")

    ax = self._fig.add_subplot(122)
    plt.title("Local Trajectory")
    traj = np.asarray(self._position_history).reshape((-1, 3))
    traj_len = traj.shape[0]

    ax.scatter(traj[max([0, traj_len-20]):, 0], traj[max([0, traj_len-20]):, 2], s=10, c='blue', facecolor=None)
    ax.set_aspect("equal")
    ax.set_adjustable("datalim")
    xlims, ylims = ax.get_xlim(), ax.get_ylim()
    ylims = (ylims[0], ylims[1]+10)

    # Draw landmarks in map
    ax.scatter(landmarks_3d[:, 0], landmarks_3d[:, 2], s=2, c='green', facecolor=None)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    self._fig.canvas.draw()
    im_vis = np.fromstring(self._fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_vis = im_vis.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)
    cv2.imshow("Visualization", im_vis)
    cv2.waitKey(1)



  # def plot_img(self, img, tag='img', store=True):
  #   pil_img = Image.fromarray( np.uint8(img) ,'RGB')
  #   pil_img.save(self.p + '/' + tag + '.png')
