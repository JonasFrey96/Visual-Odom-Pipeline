from scipy.optimize import least_squares
import numpy as np
import time
import matplotlib.pyplot as plt
from cv2 import Rodrigues
from scipy.sparse import lil_matrix

class BundleAdjuster():
    def __init__(self, ftol=1e-4, method='trf', verbosity=2, window_size=3):
        self._ftol = ftol
        self._method = method
        self._verbosity = verbosity
        self._window_size = window_size

    def _nonlinear_objective(self, x0, landmarks_kp, K):
        """
        Compute the total re-projection error using the last <window_size> frames.
        x0 contains landmark coordinates in XYZ, and camera poses <rvec, tvec>

        :param x0: vector of dimension (n_landmarks*3 + window_size*6)
        :param landmarks_kp: list of landmark keypoints
        :param K: 3x3 camera matrix
        :return: vector of dimension (n_landmarks*2*window_size vector) with the (x, y) differences
        """
        pixel_diffs = np.zeros((len(landmarks_kp)*self._window_size))

        P = x0[:len(landmarks_kp)*3].reshape((-1, 3))
        C = x0[len(landmarks_kp)*3:].reshape((-1, 6))
        for i in range(self._window_size):
            # Reconstruct transformation matrix from pose vector
            H_i = np.zeros((4, 4))
            H_i[:3, :3], _ = Rodrigues(C[i, :3])
            H_i[:3, 3] = C[i, 3:].reshape((3,))

            # Project landmarks using H and landmarks
            kp_proj = self._project(K, H_i, P)

            # Assemble pixels
            kp = np.array([k.uv_history[len(k.uv_history)-1-i].T for k in landmarks_kp]).reshape((-1, 2))

            pixel_diffs[i*len(landmarks_kp):(i+1)*len(landmarks_kp)] = np.linalg.norm(kp_proj-kp, axis=1)

        return pixel_diffs


    def _project(self, K, H, P):
        """
        Project 3D points using a camera matrix and transformation matrix.
        :param K: 3x3 camera matrix
        :param H: 4x4 camera pose transformation matrix
        :param P: Nx3 matrix of landmark 3D positions
        :return: vector of dimensions N*2. Rows alternate x and y coordinates.
        """

        # Construct Projection Matrix, Homogenize P
        M = K @ H[:3, :]
        P_homo = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)

        # Projection, De-homogenization
        P_new_homo = (M @ P_homo.T).T
        return (P_new_homo / P_new_homo[:, 2:3])[:, :2]

    def _jacobian_sparsity(self, num_landmarks):
        """
        Assuming we only optimize over camera poses and landmarks..
        x0 has dimension (n_landmarks*3 + n_poses*6), so..

        Return an (n_landmarks*window_size*2) x (n_landmarks*3 + n_poses*6)
        matrix indicating which optimization parameters affect which keypoints

        first (n_landmarks*2) rows correspond to keypoints in the earliest timestep.
        :return:
        """
        m = num_landmarks*self._window_size
        n = num_landmarks*3 + self._window_size*6
        A = lil_matrix((m, n), dtype=int)

        for i in range(num_landmarks):
            for j in range(self._window_size):
                A[j*num_landmarks+i, 3*i:3*i+3] = 1

        for i in range(self._window_size):
            A[:num_landmarks*i, 3*num_landmarks + 6*i:6*i+6] = 1

        return A


    def adjust(self, K, state, max_err_reproj=4.0):
        """Bundle adjust the camera poses, and 3D landmarks.
        # TODO: Also adjust the keypoints?
        from the <window_size> most recent frames."""

        # Filter the landmarks - only use those that have been tracked over the adjustment window
        landmarks, landmarks_kp, landmarks_unused = [], [], []
        for i in range(len(state._landmarks)):
            l_kp = state._landmarks_kp[i]
            if len(l_kp.uv_history) >= self._window_size:
                landmarks_kp.append(l_kp)
                landmarks.append(state._landmarks[i])
            else:
                landmarks_unused.append(state._landmarks[i])

        if len(landmarks) > 0:
            n_landmarks = len(landmarks)

            # Construct x0
            x0 = np.zeros((3*n_landmarks + 6*self._window_size))
            for i, l in enumerate(landmarks):
                x0[3*i:3*i+3] = l.p.reshape((3,))

            for i in range(self._window_size):
                H = state._trajectory[len(state._trajectory)-1-i]
                rvec, _ = Rodrigues(H[:3, :3])
                tvec = H[:3, 3]
                x0[3*n_landmarks+6*i: 3*n_landmarks+6*i+3] = rvec.reshape((3,))
                x0[3*n_landmarks+6*i+3: 3*n_landmarks+6*i+6] = tvec.reshape((3,))

            # Jacobian Sparsity
            A = self._jacobian_sparsity(n_landmarks)

            # loss_0 = np.sum(np.power(self._nonlinear_objective(x0, landmarks_kp, K), 2))/len(landmarks_kp)
            res = least_squares(self._nonlinear_objective, x0, jac_sparsity=A,
                                verbose=self._verbosity, x_scale='jac',
                                ftol=self._ftol, method=self._method,
                                args=(landmarks_kp, K))
            # loss_1 = res.cost
            # Build output
            landmarks_filtered, landmarks_kp_filtered = [], []
            for i in range(len(landmarks)):
                # P_old = landmarks[i].p.copy()
                # P_new = res.x[3 * i:3 * i + 3].reshape((3, 1))
                landmarks[i].p = res.x[3 * i:3 * i + 3].reshape((3, 1))

            for i in range(self._window_size):
                H_i = np.eye(4)
                H_i[:3, :3], _ = Rodrigues(res.x[n_landmarks*3+6*i:n_landmarks*3+6*i+3])
                H_i[:3, 3] = res.x[n_landmarks*3+6*i+3:n_landmarks*3+6*i+6].reshape((3,))
                state._trajectory._poses[len(state._trajectory)-1-i] = H_i

            return state
        else:
            return state
