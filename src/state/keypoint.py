from dataclasses import dataclass
import numpy as np

@dataclass
class Keypoint:
    """Class for keeping track of an item in inventory.
    feature point (uv) and the descriptor (desc) corresponds to the 
    latest *keyframe* (t_latest) frame at which the feature point was detected.

    3D point (p) is in the frame of the initial image (t0)"""
    t_first: int
    t_latest: int
    t_total:  int
    uv: np.array
    p: np.array
    des: np.array


if __name__ == "__main__":
    kp = Keypoint
    kp.t_first = 0 
    kp.t_latest = 0 
    kp.t_total = 0 
    kp.uv = np.array( [12,123])
    kp.p = np.array( [12,123,1])
    kp.des = np.zeros( (128))
    print('done')
    print(kp.t_first)
