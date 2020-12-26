from dataclasses import dataclass
import numpy as np

@dataclass
class Landmark:
    """
    3D Landmark object
    p: 3x1 np array
    des: 128x1 np array
    """
    t_latest: int
    p: np.array
    des: np.array


if __name__ == "__main__":
    l = Landmark
    l.t_latest = 0
    l.p = np.array( [12,123,1])
    l.des = np.zeros( (128))
    print('done')
