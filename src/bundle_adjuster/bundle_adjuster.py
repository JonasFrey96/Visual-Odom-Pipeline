from scipy.optimize import least_squares
import numpy as np
import time
import matplotlib.pyplot as plt

class BundleAdjuster():
    def __init__(self, ftol=1e-4, method='trf', verbosity=2):
        self._ftol = ftol
        self._method = method
        self._verbosity = verbosity

    def adjust(self, state, window_size=3):
        """Bundle adjust the camera poses, landmarks, and landmark kp's
        from the <window_size> most recent frames."""
        return state
