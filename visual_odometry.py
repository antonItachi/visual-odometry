import numpy as np
import cv2

class VisualOdometry:
    @staticmethod
    def essential_matrix(K, F):
        """
        Finding the Essential matrix from fundamental matrix.

        :param K: calibration matrix or extrinsic matrix
        :param F: fundamental matrix

        :return: Essential matrix
        """
        E = K.T @ F @ K
        U, S, V = np.linalg.svd(E)
        S = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        E = U @ S @ V
        return E

    @staticmethod
    def camera_pose(K, E):
        """
        Knowning essential matrix we can find Rotation and Translation.
        Finding 4 possible camera poses(camera orientation vector).

        :param K: calibration matrix or extrinsic matrix
        :param E: essential matrix

        :return: dictionary of camera poses (Rotation, Translation and Projetion matrix)
        """
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        U, S, V = np.linalg.svd(E)
        poses = {}

        poses["C1"] = U[:, 2].reshape(3, 1)
        poses["C2"] = -U[:, 2].reshape(3, 1)
        poses["C3"] = U[:, 2].reshape(3, 1)
        poses["C4"] = -U[:, 2].reshape(3, 1)
        poses["R1"] = U @ W @ V
        poses["R2"] = U @ W @ V
        poses["R3"] = U @ W.T @ V
        poses["R4"] = U @ W.T @ V

        for i in range(4):
            C = poses['C' + str(i + 1)]
            R = poses['R' + str(i + 1)]
            if np.linalg.det(R) < 0:
                C = -C
                R = -R
                poses['C' + str(i + 1)] = C
                poses['R' + str(i + 1)] = R
            I = np.eye(3, 3)
            M = np.hstack((I, C.reshape(3, 1)))
            poses['P' + str(i + 1)] = K @ R @ M
        return poses

    @staticmethod
    def triangulatePoints(R, T, points1, points2, K):
        """
        Create projection matrixes and Triangulates the 3d position of 2d correspondences between several images.

        :param R: Rotation matrix.
        :param T: Translation matrix.
        :param points1: Corresponding points of 1st image.
        :param points2: Corresponding points of 2nd image.
        :param K: Camera matrix.

        :return: Array of 3D points.
        """

        # Creating Projection matrixes
        C1 = np.array([[0], [0], [0]])
        R1 = np.eye(3, 3)
        my_P1 = K @ np.hstack((R1, -R1 @ C1))
        my_P2 = K @ np.hstack((R, -R @ T))

        X = []
        for i in range(points1.shape[0]):  # Iterating over all points
            triangulated_point = cv2.triangulatePoints(my_P1[:3], my_P2[:3], points1[i, :2],
                                                       points2[i, :2])
            triangulated_point = triangulated_point/triangulated_point[3]
            X.append(triangulated_point)
        return np.array(X)

    def ambiguity_matrix(self, R1, t, pt, pt_, K):
        """
        The point is to select the best variation of poses, which we got from the camera poses.
        Selection is made based on the count of 3D points.

        :param R1: Rotation matrix of each pose
        :param t: Translation matrix of each pose
        :param pt: Corresponding points of 1st image.
        :param pt_: Corresponding points of 2nd image.
        :param K: Camera matrix.

        :return: Number of points for four possible poses.
        """
        X1 = self.triangulatePoints(R1, t, pt, pt_, K)
        count = 0
        for i in range(X1.shape[0]):
            x = X1[i, :].reshape(-1, 1)
            if R1[2] @ np.subtract(x[0:3], t) > 0 and x[2] > 0:
                count += 1
        return count

    @staticmethod
    def perspective_n_point(objectPoints, imagePoints, cameraMatrix, pointsNumber=400):
        """
        The function computes projections of 3D points to the image plane
        given intrinsic and extrinsic camera parameters.

        Referring to OpenCV documentation https://amroamroamro.github.io/mexopencv/matlab/cv.solvePnPRansac.html:
        :param objectPoints: Shape of 3D point of an object is (N, 3, 1), where N is the number of points
        :param imagePoints: Shape of 2D point of an object is (N, 2, 1), where N is the number of points
        :param cameraMatrix:  Input camera matrix A = [fx 0 cx; 0 fy cy; 0 0 1].
        :param pointsNumber: Number of points.
        :return: [rvec, tvec, success, inliers]
        """
        # Reshape 3d point
        reshaped_X = [point[:-1] / point[-1] for point in objectPoints]
        objectPoints = np.array(reshaped_X)

        # Reshape 2d point
        imagePoints = imagePoints.reshape((pointsNumber, 2, 1))

        # Check shape of params if needed
        # print(objectPoints.shape)
        # print(imagePoints.shape)

        retval_ras, rvec_ras, tvec_ras, inliers = cv2.solvePnPRansac(objectPoints, imagePoints, cameraMatrix,
                                                                     distCoeffs=None, flags=cv2.SOLVEPNP_ITERATIVE,
                                                                     confidence=0.9999, reprojectionError=3)
        return rvec_ras, tvec_ras

    @staticmethod
    def reprojection_error(objectPoints, rvec, tvec, points2, K, dist_coeffs=None):
        """
        Computes the reprojection error by comparing the observed image points (points)
        with the projected image points (repro_points)

        :param objectPoints: 3D point of an abject
        :param rvec: Rotation vector from a PnP
        :param tvec: Translation vector from a PnP
        :param points2: Keypoints of the 2nd image
        :param K: Calibration matrix
        :param dist_coeffs: Distortion coefficients

        :return:
        Projected 2D image points,
        Mean squared error,
        Euclidean distance between the pair of correspondence points.
        """

        # Homogeneous to Euclidean
        objectPoints = [point[:-1] / point[-1] for point in objectPoints]
        objectPoints = np.array(objectPoints)

        repro_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, K, dist_coeffs)

        squeezed_points = repro_points.squeeze()
        points = points2.squeeze()

        # Mean error value between keypoints and PnP points.
        error = np.mean(np.sqrt(np.sum((points - squeezed_points) ** 2, axis=-1)))

        # Distance difference between correspondence keypoint and PnP point.
        # We can also calculate mean distance, but I prefer to obtain full information for further anylize
        diff = (points - squeezed_points).ravel()

        return repro_points, error, diff


