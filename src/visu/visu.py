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
  def __init__(self, K, path=None, name=None, headless=False):
    if path is None:
      path = os.path.join ( os.getcwd() , 'results')
      if name is not None: 
        path = os.path.join ( path , name)
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
    self._iter = 0
    self._headless = headless
    self._fig = plt.figure(figsize=(12, 6))

  def within_image(self, uv, img_dims):
    """Given xy pixel coordinate and the img shape (width, height),
    return boolean indicating whether the pixel lies inside the image bounds"""
    return 0 <= uv[0] <= img_dims[0] and 0 <= uv[1] <= img_dims[1]

  def update(self, im, state, landmarks_dead):
    """
    Update the state of the visualizer. K and pose are used to project
    the landmarks into the current image. Convert both landmark list and kp
    list to a list of pixel coordinates.
    Indices of the landmarks observed in the current frame are needed
    to avoid drawing out-of-frame landmarks.
    """
    # Update image
    self._im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Store keypoints
    self._landmarks_kp_px = np.asarray([kp.uv for kp in state._landmarks_kp]).reshape((-1, 2))
    self._candidates_kp_px = np.asarray([kp.uv for kp in state._candidates_kp]).reshape((-1, 2))
    self._landmark_history.append(self._landmarks_kp_px.shape[0])

    # Store trajectory
    self._position_history = []
    for i in range(len(state._trajectory)):
      H = state._trajectory[i]
      position = H[:3, :3].T @ (-H[:3, 3].reshape((-1, 1)))
      self._position_history.append(position)

    # Store active and inactive landmarks
    self._landmarks_3d = np.asarray([l.p.T for l in state._landmarks + landmarks_dead]).reshape((-1, 3))

    # Store pose for projecting landmarks
    self._H_latest = H

  def render(self):
    self._fig = plt.figure(figsize=(12, 6))
    ax = self._fig.add_subplot(221)
    plt.title("Landmarks and Keypoints")
    ax.imshow(self._im)

    # Draw keypoints
    ax.scatter(self._landmarks_kp_px[:, 0], self._landmarks_kp_px[:, 1], s=2, c='green', facecolor=None)
    ax.scatter(self._candidates_kp_px[:, 0], self._candidates_kp_px[:, 1], s=2, c='red', facecolor=None)
    ax.set_xlim([0, self._im.shape[1]])
    ax.set_ylim([self._im.shape[0], 0])
    ax.legend(["Landmarks", "Candidates"], loc='lower right')
    plt.xticks([])
    plt.yticks([])

    # Plot Landmark history
    ax = self._fig.add_subplot(245)
    plt.title("# Landmarks")
    ax.plot(list(range(len(self._landmark_history))), self._landmark_history)

    # Draw trajectory
    traj = np.asarray(self._position_history).reshape((-1, 3))
    traj_len = traj.shape[0]
    ax = self._fig.add_subplot(246)
    plt.title("Global Trajectory")
    ax.scatter(traj[:, 0], traj[:, 2], s=10, c='blue', facecolor=None)
    ax.set_aspect("equal")
    ax.set_adjustable("datalim")
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # Draw landmarks in map
    ax.scatter(self._landmarks_3d[:, 0], self._landmarks_3d[:, 2], s=0.5, c='green', facecolor=None, alpha=0.05)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax = self._fig.add_subplot(122)
    plt.title("Local Trajectory")
    ax.scatter(traj[max([0, traj_len-20]):, 0], traj[max([0, traj_len-20]):, 2], s=10, c='blue', facecolor=None)
    ax.set_aspect("equal")
    ax.set_adjustable("datalim")

    self._fig.canvas.draw()
    im_vis = np.fromstring(self._fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    im_vis = im_vis.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
    im_vis = cv2.cvtColor(im_vis, cv2.COLOR_RGB2BGR)

    if not self._headless:
      
      cv2.imshow("Visualization", im_vis)
    
    idx = str(self._iter).zfill(6)
    self._fig.savefig(os.path.join( self._p,f'out_{idx}.png'), dpi=150)
    self._iter += 1
    cv2.waitKey(1)
    
    plt.close( self._fig )

  # def plot_img(self, img, tag='img', store=True):
  #   pil_img = Image.fromarray( np.uint8(img) ,'RGB')
  #   pil_img.save(self.p + '/' + tag + '.png')
