import numpy as np

from feature_extraction import find_fundamental_matrix
from imagedataset import ImageDataset
from visual_odometry import VisualOdometry
from utils import *


MATCHES_NUMBER = 200


def main():
    image_ds = ImageDataset('../../Data/KITTI/greyscale/sequences/00/image_0',
                            '../../Data/KITTI/greyscale/sequences/00/')
    K = image_ds.camera_matrix
    vo = VisualOdometry()

    for i in range(MATCHES_NUMBER):
        # Getting a pair of images
        image_pair = image_ds.get_images()
        image1, image2 = image_pair

        # Feature mathcing and obtaning F matrix
        f1_sift, points1, points2 = find_fundamental_matrix(image1, image2, method="SIFT", f_estimate="RANSAC",
                                                            matches_number=MATCHES_NUMBER)
        points1_ = np.squeeze(points1)
        points2_ = np.squeeze(points2)

        E = vo.essential_matrix(K, f1_sift)

        pos = vo.camera_pose(K, E)

        flag = 0
        for p in range(4):
            R = pos['R' + str(p + 1)]
            T = pos['C' + str(p + 1)]

            x3d = vo.triangulatePoints(R, T, points1_, points2_, K)
            count = vo.ambiguity_matrix(R, T, points1_, points2_, K)

            if flag < count:
                flag, reg = count, str(p + 1)

        R = pos['R' + reg]
        t = pos['C' + reg]
        if t[2] < 0:
            t = -t

        rvec, tvec = vo.perspective_n_point(x3d, points2, K, pointsNumber=MATCHES_NUMBER)
        rmat, _ = cv2.Rodrigues(rvec)


if __name__ == '__main__':
    main()