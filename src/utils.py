import matplotlib.pyplot as plt
import cv2


def construct_point_cloud(correspondences):
    """
    Construct a point cloud from the correspondences.

    :param correspondences: the way we should define correspondences:
        correspondences = [((x1, y1), (x2, y2)) for (x1, y1), (x2, y2) in zip(pts1.squeeze(), pts2.squeeze())]
    :return: ploting a point cloud.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for point1, point2 in correspondences:
        # Obtain coordinates of image points
        x1, y1 = point1
        x2, y2 = point2

        # Example of simplify converting to 3d coordinates by defining Z value equal 0.
        # Z is how far away the object is located from us/camera.
        z = 0  # By default, we are asuming that the coordinates located on the same plane.

        # Visualize point in the space.
        ax.scatter(x1, y1, z, c='r', marker='o')  # red color
        ax.scatter(x2, y2, z, c='b', marker='o')  # blue color
        ax.plot([x1, x2], [y1, y2], [z, z], c='g')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def visualize_porjectionPoints(image_points, image):
    for image_point in image_points:
        x, y = image_point.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    cv2.imshow('Image with projected points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_keypoint_comparison(image_points, points, image):
    for image_point in image_points:
        x, y = image_point.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

    for point in points:
        x, y = point.ravel()
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)

    cv2.imshow('Image with projected points', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_triangulated_points(objectPoints):
    for key in objectPoints.keys():
        plt.figure(figsize=(8, 6))
        plt.scatter(objectPoints[key][:, 0], objectPoints[key][:, 2], s=5)
        plt.title(f"Triangulated points for {key+1} camera pose.")
        plt.xlabel("X")
        plt.ylabel("Z")
        # plt.grid(True)
        plt.show()