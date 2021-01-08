import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from copy import deepcopy

class TriangulatorNL():
    def __init__(self, ftol=1e-4, xtol=1e-1, method='trf', verbosity=2, loss='huber'):
        self._ftol = ftol
        self._xtol = xtol
        self._method = method
        self._verbosity = verbosity
        self._loss = loss

    def _nonlinear_objective(self, x0, K, H0, H1):
        """
        Compute the re-projection error.

        The optimization objective x0 has dimension (n_landmarks*(3+4)).
        Format is [X0 Y0 Z0, X1 Y1 Z1 ..., XN YN ZN, x0 y0 x0' y0', ... xN yN xN' yN']
        Xi Yi Zi are the landmark 3D coords, while xi yi xi' yi' are the
        pixel coordinates that were used to triangulate the landmark.
        """
        n_landmarks = len(x0)//7
        P = x0[:n_landmarks*3].reshape((-1, 3))
        x_kp = x0[n_landmarks*3:].reshape((-1, 4))
        x_proj = self._project_dual(P, K, H0, H1)
        diffs = x_kp - x_proj # Component-wise differences
        return (np.linalg.norm(diffs[:, :2], axis=1) + np.linalg.norm(diffs[:, 2:], axis=1))/2

    def _project_dual(self, P, K, H0, H1):
        return np.hstack([self._project(K, H0, P), self._project(K, H1, P)])

    def _project(self, K, H, P):
        """
        Project 3D points using a camera matrix and transformation matrix.
        :param K: 3x3 camera matrix
        :param H: 4x4 camera pose transformation matrix
        :param P: Nx3 matrix of landmark 3D positions
        :return: matrix has dim Nx2
        """
        # Construct Projection Matrix, Homogenize P
        M = K @ H[:3, :]
        P_homo = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)

        # Projection, De-homogenization
        P_new_homo = (M @ P_homo.T).T
        return (P_new_homo / P_new_homo[:, 2:3])[:, :2]

    def _jacobian_sparsity(self, num_landmarks):
        """
        We optimize the landmarks as well as their keypoints, so
        x0 has dimension (n_landmarks*(3+4))

        Return an (n_landmarks) x (n_landmarks*(3+4))
        matrix indicating which optimization parameters affect which keypoints

        Each row corresponds to the re-projection error term for a single
        landmark.

        Each (n_landmarks*3) columns correspond to landmark XYZ.
        Afterwards, alternates (x0,y0,x1,y1).
        :return:
        """
        m = num_landmarks
        n = num_landmarks*(3+4)
        A = lil_matrix((m, n), dtype=int)

        for i in range(num_landmarks):
            A[i, i*3:i*3+3] = 1
            A[(num_landmarks*3)+(4*i):(num_landmarks*3)+(4*i+4)] = 1

        return A

    def _transform_landmarks(self, landmarks, H):
        landmarks_out = []
        for l in landmarks:
            landmarks_out.append(deepcopy(l))
            landmarks_out[-1].p = ((H @ np.concatenate([landmarks_out[-1].p, np.ones((1, 1))], axis=0))[:3]).reshape((3, 1))
        return landmarks_out

    def refine(self, K, landmarks, H0, H1, keyp0, keyp1, max_err_reproj=4.0):
        """Non-linearly refine the results of a triangulation.
        Remove landmarks with high reprojection error."""

        # Discard landmarks that are behind the camera (Z < 0)
        landmarks_1 = self._transform_landmarks(landmarks, H1)
        landmarks_new, keyp0_new, keyp1_new = [], [], []
        for i, l in enumerate(landmarks_1):
            if l.p[2] > 0:
                landmarks_new.append(landmarks[i])
                keyp0_new.append(keyp0[i])
                keyp1_new.append(keyp1[i])
        landmarks = landmarks_new
        keyp0 = keyp0_new
        keyp1 = keyp1_new
        n_landmarks_init = len(landmarks)

        # Construct x0
        x0 = np.zeros(n_landmarks_init*(3+4))
        for i, (l, kp0, kp1) in enumerate(zip(landmarks, keyp0, keyp1)):
            x0[3*i:3*i+3] = l.p.reshape((3,))
            x0[(n_landmarks_init*3)+(4*i):(n_landmarks_init*3)+(4*i+2)] = kp0.uv.reshape((2,))
            x0[(n_landmarks_init*3)+(4*i+2):(n_landmarks_init*3)+(4*i+4)] = kp1.uv.reshape((2,))

        # Discard landmarks with high re-projection error
        f0 = self._nonlinear_objective(x0, K, H0, H1)
        good = f0 < max_err_reproj
        landmarks = [landmarks[i] for i in range(n_landmarks_init) if good[i]]
        keyp0 = [keyp0[i] for i in range(n_landmarks_init) if good[i]]
        keyp1 = [keyp1[i] for i in range(n_landmarks_init) if good[i]]

        # Jacobian Sparsity
        n_landmarks = len(landmarks)
        if n_landmarks:
            A = self._jacobian_sparsity(n_landmarks)

            # Construct x0
            x0 = np.zeros(n_landmarks*(3+4))
            for i, (l, kp0, kp1) in enumerate(zip(landmarks, keyp0, keyp1)):
                x0[3*i:3*i+3] = l.p.reshape((3,))
                x0[(n_landmarks*3)+(4*i):(n_landmarks*3)+(4*i+2)] = kp0.uv.reshape((2,))
                x0[(n_landmarks*3)+(4*i+2):(n_landmarks*3)+(4*i+4)] = kp1.uv.reshape((2,))

            res = least_squares(self._nonlinear_objective, x0, jac_sparsity=A,
                                verbose=self._verbosity, x_scale='jac',
                                ftol=self._ftol, method=self._method,
                                xtol=self._xtol,
                                loss=self._loss,
                                args=(K, H0, H1))

            # Build output
            for i in range(n_landmarks):
                landmarks[i].p = res.x[3 * i:3 * i + 3].reshape((3, 1))
                keyp0[i].uv = res.x[(n_landmarks*3)+(4*i):(n_landmarks*3)+(4*i+2)].reshape((2, 1))
                keyp1[i].uv = res.x[(n_landmarks*3)+(4*i+2):(n_landmarks*3)+(4*i+4)].reshape((2, 1))

            # Remove points with high re-projection error
            good = res.fun < max_err_reproj
            landmarks = [landmarks[i] for i in range(n_landmarks) if good[i]]
            keyp0 = [keyp0[i] for i in range(n_landmarks) if good[i]]
            keyp1 = [keyp1[i] for i in range(n_landmarks) if good[i]]

            return landmarks, keyp0, keyp1
        else:
            return [], [], []


"""https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py"""
# Initialize consts to be used in iterative_LS_triangulation()
iterative_LS_triangulation_C = -np.eye(2, 3)

def triangulatePoints_ILS(P_0, P_1, uv0, uv1, tolerance=3.e-5):
    """
    Iterative (Linear) Least Squares based triangulation.
    From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
    Relative speed: 0.025

    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    "tolerance" is the depth convergence tolerance.

    Additionally returns a status-vector to indicate outliers:
        1: inlier, and in front of both cameras
        0: outlier, but in front of both cameras
        -1: only in front of second camera
        -2: only in front of first camera
        -3: not in front of any camera
    Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    N = uv1.shape[0]

    # Create array of triangulated points
    x = np.empty((4, N));
    x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
    x_status = np.empty(N, dtype=int)

    # Initialize C matrices
    C0 = np.array(iterative_LS_triangulation_C)
    C1 = np.array(iterative_LS_triangulation_C)

    for xi in range(N):
        # Build C matrices, to construct A and b in a concise way
        C0[:, 2] = uv0[xi, :, :].reshape((2,))
        C1[:, 2] = uv1[xi, :, :].reshape((2,))

        # Build A matrix
        A[0:2, :] = C0.dot(P_0[0:3, 0:3])  # C1 * R1
        A[2:4, :] = C1.dot(P_1[0:3, 0:3])  # C2 * R2

        # Build b vector
        b[0:2, :] = C0.dot(P_0[0:3, 3:4])  # C1 * t1
        b[2:4, :] = C1.dot(P_1[0:3, 3:4])  # C2 * t2
        b *= -1

        # Init depths
        d0 = d1 = 1.

        for i in range(10):  # Hartley suggests 10 iterations at most
            # Solve for x vector
            # x_old = np.array(x[0:3, xi])    # TODO: remove
            cv2.solve(A, b, x[0:3, xi:xi + 1], cv2.DECOMP_SVD)

            # Calculate new depths
            d0_new = P_0[2, :].dot(x[:, xi])
            d1_new = P_1[2, :].dot(x[:, xi])

            # Convergence criterium
            # print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
            # print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
            ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
            # print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
            ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
            if abs(d0_new - d0) <= tolerance and abs(d1_new - d1) <= tolerance:
                # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
                # if abs((d1_new - d1) / d1) <= 3.e-6 and \
                # abs((d2_new - d2) / d2) <= 3.e-6: #and \
                # abs(d1_new - d1) <= tolerance and \
                # abs(d2_new - d2) <= tolerance:
                # if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
                # 1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
                # abs(d1_new - d1) <= tolerance and \    # TODO: remove
                # abs(d2_new - d2) <= tolerance:    # TODO: remove
                break

            # Re-weight A matrix and b vector with the new depths
            A[0:2, :] *= 1 / d0_new
            A[2:4, :] *= 1 / d1_new
            b[0:2, :] *= 1 / d0_new
            b[2:4, :] *= 1 / d1_new

            # Update depths
            # d_old = d1    # TODO: remove
            # d1_old = d2    # TODO: remove
            d0 = d0_new
            d1 = d1_new

        # Set status
        x_status[xi] = (i < 10 and  # points should have converged by now
                       (d0_new > 0 and d1_new > 0))  # points should be in front of both cameras
        if d0_new <= 0: x_status[xi] -= 1
        if d1_new <= 0: x_status[xi] -= 2

    return x[0:3, :].T, x_status