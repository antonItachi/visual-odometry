# Study of Visual Odometry and Structure from Motion (SFM)
  
## The fundamental insights within Pipeline
 - ### Feature extraction and Funamental matrix estimation
According to `feature_extraction.py`, you can find matching points between two images and Fundamental matrix.
   
In this function you can try several main methods - `SIFT`, `BRISK` and `ORB`. You can compare them by ploting or number of matches.
   
Then you estimate your fundamental matrix by decreasing number of inliers.
Methods, that can be used - `8-points`, `LMEDS`, `RANSAC`.
   
   Also you can choose the number of good matches. In my case thats helped me a lot to delete imbalanced points. I detected them after triangulation and PnP.
 - ### Essential matrix and Fundamental matrix
Knowing Fundamental matrix and calibration matrix(intrinsic matrix) can find Essential matrix using formula.
   <p align="center">
     <img src="https://github.com/antonItachi/visual-odometry/assets/78692457/d78ec2a8-3d7d-4d62-a737-742cb4e244c5">
   </p>

   <p align="justify", >The distinction is, that in the case of Fundamental matrix, the points are specified in pixel coordinates, while in the case of \tEssential matrix, the points are given in "normalized image coordinates"(uncalibrated). These normalized coordinates have their origin at the image's optical center, with the x and y coordinates being scaled by Fx and Fy respectively. Thats why we should know the intrinsic parameters - to normalize optical center and scale coordinates. </p>
 
 - ### Camera pose extraction from Essential matrix
By decomposing E matrix, possible to get Rotation and Translation.
   <p align="center">
     <img src="https://github.com/antonItachi/visual-odometry/assets/78692457/35fd3cee-7602-4cd9-b17b-d7dd1df77ad3">
   </p>

Knowing that Tx is [Skew-Symmetric matrix](https://en.wikipedia.org/wiki/Skew-symmetric_matrix) and R is [Orthonormal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix) it is possible to decouple R and T from their product using `SVD(Singular Value Decomposition`. In total, there are four camera pose configurations.
   
 - ### Projection matrix
The projection matrix is typically used to project 3D points in a scene onto a 2D image plane. It combines the intrinsic parameters of the camera (such as focal length and principal point) with the extrinsic parameters (such as camera position and orientation) to perform the projection.
   <p align="center">
     <img src="https://github.com/antonItachi/visual-odometry/assets/78692457/88e4c194-7f74-4e56-8663-0e65076f7c95">
  </p>
The projection matrix transforms homogeneous 3D coordinates [X,Y,Z,1] into homogeneous 2D coordinates [u,v,1] on the image plane. 

 - ### Point Triangulation
Triangulation is the process of determining a point in 3D space given its projections onto two, or more, images(two in my case). Note that, triangulation need two camera matrixes(reference one), by default it looks like this:
   <p align="center">
   <img src="https://github.com/antonItachi/visual-odometry/assets/78692457/d63ad77a-c9b6-4939-a19d-abea4e05dd3a">
   </p>
   
 - ### Camera Pose Disambiguation

Given four camera pose configuration(R and C) and their triangulated points, find the right camera pose by applying the cheirality condition i.e. the reconstructed points must be in front of the cameras. The triangulated point is in front of the camera if and only if `R3(X - C) > 0`, where `R3` is the z-axis of the rotation matrix. The best variation of camera pose is the one that has the most triangulated points satisfying the condition.

 - ### Perspective-n-Points
<p align="justify">
     This method is used to determine the position and orientation of the camera relative to the world coordinate system based on matching 3D points in the world coordinate system with their 2D projections on the image. PnP solves the problem of determining camera position based on known 3D points and their corresponding 2D projections in the image. Unlike the Essential matrix Decomposition, where you find camera pose relative to the previous position.
</p>













## Project Structure:
  
```markdown
visual-odometry/
   src/
      ├── feature_extraction # (python) functions to find corresponding points and estimate Fundamental matrix
      ├── visual_odometry # (python) class with main visual odometry functions.
      ├── utils # (python/matplotlib) utility function for data processing and visualization
      ├── imagedataset # (python) class for preproccessing `KITTI` dataset, to get the pair of images and Calibration matrix (Intrinsic matrix)

  ├── main.py # (python) script to demonstrate implemented methods
  ├── README.md
```
