from dataclasses import dataclass
import numpy as np

@dataclass
class Keypoint:
    """
    Detected Shi-Tomasi key-point object.
    Keep track of when a key-point was first observed.
    If the track length is sufficiently long, and baseline is sufficiently
    large, we triangulate this keypoint.

    uv_first: 2x1 np array
    uv: 2x1 np array (latest uv)
    des: 128x1 np array
    """
    t_first: int
    t_total: int
    uv_first: np.array
    uv: np.array
    des: np.array
    uv_history: list


if __name__ == "__main__":
    kp = Keypoint
    kp.t_first = 0
    kp.t_total = 0 
    kp.uv_first = np.array([12, 123])
    kp.uv_latest = np.array([12, 123])
    kp.des = np.zeros((128))
    print('done')
    print(kp.t_first)
