import random
import cv2
import numpy as np


def estimate_fundamental_matrix(pts1, pts2, method='8-point', num_iterations=1000):
    """
    Refine the fundamental matrix by decreasing the number of inliers thus define the best fundamental matrix,
    to verify and improve the accuracy of the matches between keypoints.

    :param pts1: key points of 1st image.
    :param pts2: correspondence key points of 2nd image.
    :param method: choose estimated method from ['8-point', 'LMDES', 'RANSAC'].
    :param num_iterations: number of iterations.

    :return: best Funamental matrix variation.
    """

    best_f = None
    best_inliers = 0

    if method == '8-point':
        method_flag = cv2.FM_8POINT
    elif method == 'LMEDS':
        method_flag = cv2.FM_LMEDS
    elif method == 'RANSAC':
        method_flag = cv2.FM_RANSAC
    else:
        raise ValueError("Unsupported method. Please choose '8-point', 'LMEDS', or 'RANSAC'.")

    for i in range(num_iterations):
        # Randomly select a subset of correspondences
        subset_indices = random.sample(range(len(pts1)), 8)
        subset_pts1 = pts1[subset_indices]
        subset_pts2 = pts2[subset_indices]

        # Estimate fundamental matrix using subset
        F, mask = cv2.findFundamentalMat(subset_pts1, subset_pts2, method_flag)

        # Calculate number of inliers
        inliers = np.sum(mask)

        # If this model is better than previous best model, update
        if inliers > best_inliers:
            best_inliers = inliers
            best_f = F

    return best_f


def find_fundamental_matrix(img1, img2, method="SIFT", f_estimate="RASNAC", matches_number=300, verbose=False):
    """
    Find the fundametnal matrix with feature extraction from two images. Fundametnal matrix represents the geometric
    relationship between corresponding points in two images taken by the camera from different viewpoints.

    :param img1: First image
    :param img2: Second image
    :param method: Method used to detect descriptors and keypoints. Possible values are ['SIFT', 'ORB', 'BRISK'].
    :param f_estimate: Method used to estimate the fundamental matrix for the function "estimate_fundamental_matrix".
    Possible values are ['8-point', 'LMDES', 'RANSAC']
    :param matches_number: Number of good matches we want to use to avoid outliers and imbalanced values.
    :param verbose: If true, visualize the images with their corresponding points and general information.

    :return: Fundamental matrix and corresponding points (F, pts1, pts2)
    """

    # Creating descriptors and keypoints for chosen method
    if method == "SIFT":
        sift = cv2.SIFT.create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

    elif method == "BRISK":
        brisk = cv2.BRISK.create()
        kp1, des1 = brisk.detectAndCompute(img1, None)
        kp2, des2 = brisk.detectAndCompute(img2, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)

    elif method == "ORB":
        orb = cv2.ORB.create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)

    else:
        raise ValueError("Unsupported method. Please choose 'SIFT', 'BRISK', or 'ORB'.")

    # Creating matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test for obtaining good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.90 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # Get coordinates of good matches
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches[:matches_number]]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches[:matches_number]]).reshape(-1, 1, 2)

    # Finding fundamental matrix using RANSAC
    F = estimate_fundamental_matrix(pts1, pts2, method=f_estimate)

    # Get some visualize and text data for logs
    if verbose:
        # Coordinates of matched data
        print("Number of matches:", len(pts1))
        print("Coordinates of matched points on image 1:")
        print(pts1[:5])
        print("Coordinates of matched points on image 2:")
        print(pts2[:5])

        # Plot matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow(f"{method} Matches with {f_estimate}", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return F, pts1, pts2