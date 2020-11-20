from dataclasses import dataclass
import numpy as np

@dataclass
class Keypoint:
    """Class for keeping track of an item in inventory."""
    t_first: int
    t_latest: int
    t_total:  int
    uv: np.array
    p: np.array