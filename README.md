## Structure From Motion


This Python script performs 3D reconstruction from a pair of stereo images using OpenCV and Open3D. The reconstruction involves camera calibration, feature matching, fundamental matrix estimation, essential matrix calculation, triangulation, and visualization of the 3D point cloud.

## Prerequisites

- Python 3
- OpenCV
- Open3D
- NumPy
- Matplotlib
- scikit-learn


## Script Components

- **calibrate_and_undistort:**
  Calibrates the camera using chessboard images and undistorts input images.

- **find_correspondence_points:**
  Finds corresponding points using SIFT feature matching.

- **calculate_fundamental_matrix:**
  Estimates the fundamental matrix using the eight-point algorithm.

- **calculate_essential_matrix:**
  Calculates the essential matrix from the fundamental matrix and camera intrinsic matrix.

- **decompose_essential_matrix:**
  Decomposes the essential matrix into rotation matrices and translation vectors.

- **triangulate_with_best_projection:**
  Triangulates 3D points using the best projection matrix.

- **get_dominant_colors:**
  Retrieves dominant colors in an image using k-means clustering.

- **save_pcd_to_file:**
  Saves 3D points and colors to a PCD file.

- **visualize_3d_point_cloud:**
  Visualizes the 3D point cloud using Open3D.
