from scipy.optimize import least_squares
import numpy as np
from cv2 import Rodrigues
from scipy.sparse import lil_matrix
import logging

class BundleAdjuster():
    def __init__(self, xtol=1e-3, ftol=1e-4, method='trf', verbosity=2, loss='huber',
                 window_size=3, max_err_reproj=2.0):
        self._ftol = ftol
        self._xtol = xtol
        self._method = method
        self._verbosity = verbosity
        self._window_size = window_size
        self._max_err_reproj = max_err_reproj
        self._loss = loss

    def _nonlinear_objective(self, x0, landmarks_kp, K, num_poses):
        """
        Compute the total re-projection error using the last <window_size> frames.
        x0 contains landmark coordinates in XYZ, and camera poses <rvec, tvec>

        :param x0: vector of dimension (n_landmarks*3 + window_size*6)
        :param landmarks_kp: list of landmark keypoints
        :param K: 3x3 camera matrix
        :return: vector of dimension (n_landmarks*2*window_size vector) with the (x, y) differences
        """
        pixel_diffs = np.zeros(len(landmarks_kp)*num_poses*2)

        P = x0[:len(landmarks_kp)*3].reshape((-1, 3))
        C = x0[len(landmarks_kp)*3:].reshape((-1, 6))
        for i in range(num_poses):
            # Reconstruct transformation matrix from pose vector
            H_i = np.zeros((4, 4))
            H_i[:3, :3], _ = Rodrigues(C[i, :3])
            H_i[:3, 3] = C[i, 3:].reshape((3,))

            # Project landmarks using H and landmarks
            kp_proj = self._project(K, H_i, P)

            # Assemble pixels
            kp = np.array([k.uv_history[len(k.uv_history)-1-i].T for k in landmarks_kp]).reshape((-1,))

            pixel_diffs[2*i*len(landmarks_kp):2*(i+1)*len(landmarks_kp)] = (kp_proj-kp).reshape((-1,))

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
        return (P_new_homo / P_new_homo[:, 2:3])[:, :2].reshape((-1,))

    def _jacobian_sparsity(self, num_landmarks, num_poses):
        """
        Assuming we only optimize over camera poses and landmarks..
        x0 has dimension (n_landmarks*3 + n_poses*6), so..

        Return an (n_landmarks*window_size*2) x (n_landmarks*3 + n_poses*6)
        matrix indicating which optimization parameters affect which keypoints

        first (n_landmarks*2) rows correspond to keypoints in the latest timestep.
        Rows are alternating X, Y differences.
        :return:
        """
        m = num_landmarks*num_poses*2
        n = num_landmarks*3 + num_poses*6
        A = lil_matrix((m, n), dtype=int)

        # TODO: check this
        for i in range(num_landmarks):
            for j in range(num_poses):
                # A[j*num_landmarks*2 + 2*i: j*num_landmarks*2 + 2*i + 2, 3*i: 3*i+3] = 1
                A[2*(j*num_landmarks+i):2*(j*num_landmarks+i)+2, 3*i:3*i+3] = 1

        for i in range(num_poses):
            A[(2*num_landmarks*i):(2*num_landmarks*(i+1)), (3*num_landmarks)+(6*i):(3*num_landmarks)+(6*i)+6] = 1

        return A


    def adjust(self, K, state):
        """Bundle adjust the camera poses, and 3D landmarks.
        # TODO: Also adjust the keypoints?
        from the <window_size> most recent frames."""

        # Filter the landmarks - only use those that have been tracked over the adjustment window

        landmarks, landmarks_kp, landmarks_unused, landmarks_kp_unused = [], [], [], []
        for i in range(len(state._landmarks)):
            l_kp = state._landmarks_kp[i]
            if len(l_kp.uv_history) >= self._window_size:
                landmarks_kp.append(l_kp)
                landmarks.append(state._landmarks[i])
            else:
                landmarks_kp_unused.append(l_kp)
                landmarks_unused.append(state._landmarks[i])

        if len(landmarks) > 0:
            n_landmarks = len(landmarks)
            logging.info(f"Bundle adjusting {n_landmarks} landmarks over last {self._window_size} frames")

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

            # # Discard points with high re-projection error (in the latest frame)
            # n_landmarks_init = len(landmarks)
            # f0 = self._nonlinear_objective(x0, landmarks_kp, K).reshape((-1, 2))
            # f0 = np.linalg.norm(f0[:n_landmarks, :], axis=1)
            #
            # good = f0 < self._max_err_reproj
            # landmarks = [landmarks[i] for i in range(n_landmarks_init)
            #              if good[i]]
            # landmarks_kp = [landmarks_kp[i] for i in range(n_landmarks_init)
            #                 if good[i]]
            n_landmarks = len(landmarks)

            if len(landmarks):
                # *Re*-Construct x0
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
                A = self._jacobian_sparsity(n_landmarks, num_poses=self._window_size)

                if self._method == 'lm':
                    res = least_squares(self._nonlinear_objective, x0,
                                        verbose=self._verbosity,
                                        ftol=self._ftol, method=self._method,
                                        loss='linear', xtol=self._xtol,
                                        args=(landmarks_kp, K, self._window_size))
                elif self._method == 'trf':
                    res = least_squares(self._nonlinear_objective, x0,
                                        verbose=self._verbosity,
                                        ftol=self._ftol, method=self._method,
                                        xtol=self._xtol, loss=self._loss,
                                        args=(landmarks_kp, K, self._window_size),
                                        jac_sparsity=A, x_scale='jac')

                # Extract refined landmarks and poses from optimization result
                for i in range(len(landmarks)):
                    landmarks[i].p = res.x[3 * i:3 * i + 3].reshape((3, 1))

                for i in range(self._window_size):
                    H_i = np.eye(4)
                    H_i[:3, :3], _ = Rodrigues(res.x[n_landmarks*3+6*i:n_landmarks*3+6*i+3])
                    H_i[:3, 3] = res.x[n_landmarks*3+6*i+3:n_landmarks*3+6*i+6].reshape((3,))
                    state._trajectory._poses[len(state._trajectory)-1-i] = H_i

        state._landmarks = landmarks+landmarks_unused
        state._landmarks_kp = landmarks_kp+landmarks_kp_unused
        return state
