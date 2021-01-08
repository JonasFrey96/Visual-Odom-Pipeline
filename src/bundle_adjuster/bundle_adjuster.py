from scipy.optimize import least_squares
import numpy as np
from cv2 import Rodrigues
from scipy.sparse import lil_matrix
import logging

class BundleAdjuster():
    def __init__(self, xtol=1e-2, ftol=1e-4, method='trf', verbosity=2, loss='huber',
                 window_size=3, max_err_reproj=2.0):
        self._ftol = ftol
        self._xtol = xtol
        self._method = method
        self._verbosity = verbosity
        self._window_size = window_size
        self._max_err_reproj = max_err_reproj
        self._loss = loss

    def _nonlinear_objective(self, x0, landmarks_kp, landmarks, observed_landmarks, K, t_now):
        """
        Compute the re-projection error of landmarks over the last <num_poses>
        Compute the total re-projection error using the last <window_size> frames.
        x0 contains landmark coordinates in XYZ, and camera poses <rvec, tvec>

        :param x0: vector of dimension (n_landmarks*3 + window_size*6)
        :param landmarks_kp: list of landmark keypoints
        :param K: 3x3 camera matrix
        :return: ?
        """

        # Construct pixel_diffs
        # Dimension of pixel_diffs is (Num pixels observed at each time in the window)
        pixel_diffs = []
        n_landmarks = len(landmarks_kp)

        P = x0[:n_landmarks*3].reshape((-1, 3))
        C = x0[n_landmarks*3:].reshape((-1, 6))
        for i in range(self._window_size):
            # Get landmark subset that were observed at time (t_now-i)
            landmarks_subset, landmarks_kp_subset, P_subset = [], [], []
            for j in observed_landmarks[i]:
                P_subset.append(P[j, :])
                landmarks_kp_subset.append(landmarks_kp[j])
                landmarks_subset.append(landmarks[j])
            P_subset = np.array(P_subset).reshape((-1, 3))

            # Reconstruct transformation matrix from pose vector
            H_i = np.zeros((4, 4))
            H_i[:3, :3], _ = Rodrigues(C[i, :3])
            H_i[:3, 3] = C[i, 3:].reshape((3,))

            # Project landmarks using H and landmarks
            kp_proj = self._project(K, H_i, P_subset)

            # Assemble pixels
            kp = []
            for j, k in enumerate(landmarks_kp_subset):
                l = landmarks_subset[j]
                history_index = (t_now-i)-landmarks_subset[j].t_latest + (len(k.uv_history)-1)
                kp.append(k.uv_history[history_index])
            kp = np.array(kp).reshape((-1, 2))

            # Determine index offset
            pixel_diffs.append(np.linalg.norm(kp_proj-kp, axis=1, ord=2).reshape((-1, 1)))

        return np.vstack(pixel_diffs).reshape((-1,)).astype(np.float64) # Check dimension


    def _project(self, K, H, P):
        """
        Project 3D points using a camera matrix and transformation matrix.
        :param K: 3x3 camera matrix
        :param H: 4x4 camera pose transformation matrix
        :param P: Nx3 matrix of landmark 3D positions
        :return: matrix of dimensions (N, 2). Rows alternate x and y coordinates.
        """

        # Construct Projection Matrix, Homogenize P
        M = K @ H[:3, :]
        P_homo = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)

        # Projection, De-homogenization
        P_new_homo = (M @ P_homo.T).T
        return (P_new_homo / P_new_homo[:, 2:3])[:, :2]

    def _jacobian_sparsity(self, num_landmarks, observed_landmarks):
        """
        Assuming we only optimize over camera poses and landmarks..
        x0 has dimension (n_landmarks*3 + n_poses*6), so..

        Return an (n_landmarks*window_size*2) x (n_landmarks*3 + n_poses*6)
        matrix indicating which optimization parameters affect which keypoints

        first (n_landmarks*2) rows correspond to keypoints in the latest timestep.
        Rows are alternating X, Y differences.
        :param observed_landmarks a list of lists. list[i] has a list of indices describing the landmarks observed i timesteps ago.
        :return:
        """
        m = sum([len(obs) for obs in observed_landmarks])
        n = num_landmarks*3 + self._window_size*6
        A = lil_matrix((m, n), dtype=int)

        ## Old
        # for i in range(num_landmarks):
        #     for j in range(self._window_size):
        #         A[2*(j*num_landmarks+i):2*(j*num_landmarks+i)+2, 3*i:3*i+3] = 1
        #
        # for i in range(self._window_size):
        #     A[(2*num_landmarks*i):(2*num_landmarks*(i+1)), (3*num_landmarks)+(6*i):(3*num_landmarks)+(6*i)+6] = 1
        row_start, row_end = 0, 0
        for i in range(self._window_size):
            num_landmarks_observed = len(observed_landmarks[i])
            row_end += num_landmarks_observed

            # Activate derivative term for camera pose at current timestep = 1
            A[row_start:row_end, (3*num_landmarks)+(6*i):(3*num_landmarks)+(6*i)+6] = 1

            # Activate derivative term for each landmark
            for j in range(num_landmarks_observed):
                landmark_idx = observed_landmarks[i][j]
                A[row_start+j, (3*landmark_idx):(3*landmark_idx)+3] = 1

            # Shift indices to address landmarks observed in the next timestep
            row_start = row_end
        return A


    def adjust(self, state, landmarks_dead, landmarks_kp_dead, K, t_now):
        """Bundle adjust the camera poses, and 3D landmarks.
        from the <window_size> most recent frames."""

        # Determine whether each landmark was observed at each timestep
        observed_landmarks = [list() for i in range(self._window_size)]
        n_landmarks_active = len(state._landmarks)

        # Determine earliest timestamps for each of the dead landmarks
        t_earliest = []
        for i, k in enumerate(landmarks_kp_dead):
            l = landmarks_dead[i]
            t_earliest.append(l.t_latest - (len(k.uv_history)-1))

        # Filter out landmarks to be refined
        refine_landmarks, refine_landmarks_kp = state._landmarks, state._landmarks_kp
        unrefined_landmarks, unrefined_landmarks_kp = [], []
        for i, l in enumerate(landmarks_dead):
            if (t_now-t_earliest[i]) < self._window_size:
                refine_landmarks.append(l)
                refine_landmarks_kp.append(landmarks_kp_dead[i])
            else:
                unrefined_landmarks.append(l)
                unrefined_landmarks_kp.append(landmarks_kp_dead[i])

        # Determine the subset of landmarks observed at each timestep in the window
        for t in range(self._window_size):
            for j, k in enumerate(refine_landmarks_kp):
                l = refine_landmarks[j]
                history_index = (t_now-t)-l.t_latest + (len(k.uv_history)-1)
                if 0 <= history_index <= (len(k.uv_history)-1):
                    observed_landmarks[t].append(j)

        n_landmarks_refine = len(refine_landmarks)
        logging.info(f"Bundle adjusting {n_landmarks_refine} landmarks over last "
                     f"{self._window_size} frames. ({n_landmarks_active}/{n_landmarks_refine})")

        # Construct x0
        x0 = np.zeros((3*n_landmarks_refine + 6*self._window_size))
        for i, l in enumerate(refine_landmarks):
            x0[3*i:3*i+3] = l.p.reshape((3,))

        for i in range(self._window_size):
            if len(state._trajectory)-1-i < 0:
                break
            H = state._trajectory[len(state._trajectory)-1-i]
            rvec, _ = Rodrigues(H[:3, :3])
            tvec = H[:3, 3]
            x0[3*n_landmarks_refine+6*i: 3*n_landmarks_refine+6*i+3] = rvec.reshape((3,))
            x0[3*n_landmarks_refine+6*i+3: 3*n_landmarks_refine+6*i+6] = tvec.reshape((3,))

        # Jacobian Sparsity
        A = self._jacobian_sparsity(n_landmarks_refine, observed_landmarks)

        if self._method == 'lm' and A.shape[0] > A.shape[1]:
            res = least_squares(self._nonlinear_objective, x0,
                                verbose=self._verbosity,
                                ftol=self._ftol, method='lm',
                                loss='linear', xtol=self._xtol,
                                max_nfev=1000000,
                                args=(state._landmarks_kp, refine_landmarks, observed_landmarks, K, t_now))
        else:
            res = least_squares(self._nonlinear_objective, x0,
                                verbose=self._verbosity,
                                ftol=self._ftol, method=self._method,
                                xtol=self._xtol, loss=self._loss,
                                args=(state._landmarks_kp, refine_landmarks, observed_landmarks, K, t_now),
                                jac_sparsity=A, x_scale='jac')

        # Extract refined landmarks and poses from optimization result
        for i in range(n_landmarks_refine):
            if i < n_landmarks_active:
                state._landmarks[i].p = res.x[3 * i:3 * i + 3].reshape((3, 1))
            else:
                refine_landmarks[i].p = res.x[3 * i:3 * i + 3].reshape((3, 1))

        landmarks_dead = [refine_landmarks[i] for i in range(n_landmarks_active, n_landmarks_refine)] + unrefined_landmarks
        landmarks_kp_dead = [refine_landmarks_kp[i] for i in range(n_landmarks_active, n_landmarks_refine)] + unrefined_landmarks_kp

        for i in range(self._window_size):
            t = t_now-i
            if not (t in state._trajectory._poses):
                break
            H_i = np.eye(4)
            H_i[:3, :3], _ = Rodrigues(res.x[n_landmarks_refine*3+6*i:n_landmarks_refine*3+6*i+3])
            H_i[:3, 3] = res.x[n_landmarks_refine*3+6*i+3:n_landmarks_refine*3+6*i+6].reshape((3,))
            state._trajectory._poses[t] = H_i

        return state, landmarks_dead, landmarks_kp_dead
