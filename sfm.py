import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import KMeans


def calibrate_and_undistort(image_files, chessboard_size):
    # Object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in the image plane

    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print("Error loading image:", fname)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # # Draw the corners on the image and display
            # cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            # cv2.imshow('Corners', img)
            # cv2.waitKey(500)

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    re_projection_error = mean_error / len(objpoints)

    # Print and save the results
    print("Calibration Matrix:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)
    print("\nRe-projection Error:", re_projection_error)

    # Save calibration matrix
    np.savez('calibration_data.npz', mtx=mtx, dist=dist)

    # Undistort the images
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            print("Error loading image:", fname)
            continue

        # Undistort the image
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

        # # Display the undistorted image
        # cv2.imshow('Undistorted Image', undistorted_img)
        # cv2.waitKey(500)

    # cv2.destroyAllWindows()
    return mtx, dist 

def calculate_fundamental_matrix(keypoints1, keypoints2, matches, mtx, dist):
    # Extract corresponding points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # # Undistort the points using the calibration parameters
    # points1 = cv2.undistortPoints(points1.reshape(1, -1, 2), mtx, dist).reshape(-1, 2)
    # points2 = cv2.undistortPoints(points2.reshape(1, -1, 2), mtx, dist).reshape(-1, 2)

    # Normalize the points using NumPy's linalg.norm
    points1_norm = points1 / np.linalg.norm(points1, axis=1)[:, None]
    points2_norm = points2 / np.linalg.norm(points2, axis=1)[:, None]

    # Use the eight-point algorithm to compute the fundamental matrix
    F, _ = cv2.findFundamentalMat(points1_norm, points2_norm, cv2.FM_8POINT)

    # Verify the constraint: qr^T * F * ql = 0
    for i in range(len(points1_norm)):
        pt1 = np.append(points1_norm[i], 1)
        pt2 = np.append(points2_norm[i], 1)
        constraint = np.matmul(np.matmul(pt2.T, F), pt1)
    print(f'Constraint {i+1}: {constraint}')

    return F




def calculate_essential_matrix(F, K):
    # Calculate Essential matrix (E) using the camera intrinsic matrix (K)
    E = np.dot(np.dot(K.T, F), np.linalg.inv(K)).astype(np.float64)

    # Verify that the determinant of E is zero
    det_E = np.linalg.det(E)
    print(f'Determinant of E: {det_E}')

    return E



def decompose_essential_matrix(E):
    # Decompose Essential matrix into Rotation (R) and Translation (T)
    R1,R2,t = cv2.decomposeEssentialMat(E)
    return R1,R2,t

# def create_projection_matrices(K, R, T):
#     # Create the Projection matrix (P) using the Intrinsic matrix (K), Rotation matrix (R), and Translation matrix (T)
#     P = np.hstack((R, T))

#     return K @ P

def find_correspondence_points(image_path1, image_path2, mtx, dist):
    # Read the two images
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    # Draw keypoints on images with different colors
    img_keypoints1 = cv2.drawKeypoints(gray1, keypoints1, None, color=(0, 0, 255))  # Red color for the first image
    img_keypoints2 = cv2.drawKeypoints(gray2, keypoints2, None, color=(0, 255, 255))  # Yellow color for the second image
    # Draw keypoints on images
    # Resize keypoints images to fit the screen
    scale_percent_keypoints = 50  # adjust as needed
    width_keypoints = int(img_keypoints1.shape[1] * scale_percent_keypoints / 100)
    height_keypoints = int(img_keypoints1.shape[0] * scale_percent_keypoints / 100)
    dim_keypoints = (width_keypoints, height_keypoints)
    resized_img_keypoints1 = cv2.resize(img_keypoints1, dim_keypoints, interpolation=cv2.INTER_AREA)
    resized_img_keypoints2 = cv2.resize(img_keypoints2, dim_keypoints, interpolation=cv2.INTER_AREA)

    # Display images with resized SIFT keypoints
    cv2.imshow('SIFT Keypoints - Image 1', resized_img_keypoints1)
    cv2.imshow('SIFT Keypoints - Image 2', resized_img_keypoints2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    # Draw matches on images
    img_matches = cv2.drawMatches(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Resize matches image to fit the screen
    scale_percent_matches = 50  # adjust as needed
    width_matches = int(img_matches.shape[1] * scale_percent_matches / 100)
    height_matches = int(img_matches.shape[0] * scale_percent_matches / 100)
    dim_matches = (width_matches, height_matches)
    resized_img_matches = cv2.resize(img_matches, dim_matches, interpolation=cv2.INTER_AREA)

    # Display the matches with SIFT features
    cv2.imshow('SIFT Matches', resized_img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    F = calculate_fundamental_matrix(keypoints1, keypoints2, good_matches, mtx, dist)
    print("Fundamental Matrix :",F)


    # Calculate Essential matrix (E)
    E = calculate_essential_matrix(F, mtx)
    print("Essential Matrix :",E)

    # Decompose Essential matrix into Rotation (R) and Translation (T)
    R1, R2, T = cv2.decomposeEssentialMat(E)

    # Print the results
    print("Rotation Matrix R1:")
    print(R1)
    print("\nRotation Matrix R2:")
    print(R2)
    print("\nTranslation Vector T:")
    print(T)

    # Create Projection matrices for camera 1 (P0) and camera 2 (P1)
    P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = create_projection_matrices(mtx, R1, T)  # Use R1 for the first camera

    # print("\nProjection Matrix P0:")
    # print(P0)
    # print("\nProjection Matrix P1:")
    # print(P1)

    return R1,R2,T,keypoints1,keypoints2,good_matches

def create_projection_matrices(K, R, T):
    # Create the Projection matrix (P) using the Intrinsic matrix (K), Rotation matrix (R), and Translation matrix (T)
    P1 = np.hstack((R, T))
    P2 = np.hstack((R, -T))
    P3 = np.hstack((R, T))
    P4 = np.hstack((R, -T))

    return K @ P1, K @ P2, K @ P3, K @ P4

def estimate_reprojection_error_for_camera(points_3d, camera_matrix, rvec, tvec, keypoints):
    # Project the 3D points to 2D image space
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)

    # Calculate the reprojection error
    total_error = 0
    for i in range(len(points_3d)):
        # Calculate the Euclidean distance between the actual and reprojected 2D points
        error = cv2.norm(keypoints[i].pt, points_2d[i][0])
        total_error += error

    # Calculate the mean reprojection error
    mean_error = total_error / len(points_3d)

    return mean_error


def linear_ls_triangulation(u0, P0, u1, P1):
   
    u0 = np.array([u0[0], u0[1], 1.0])
    u1 = np.array([u1[0], u1[1], 1.0])
    
    A = np.zeros((4,4))
    A[0] = u0[0] * P0[2,:] - P0[0,:]
    A[1] = u0[1] * P0[2,:] - P0[1,:]
    A[2] = u1[0] * P1[2,:] - P1[0,:]
    A[3] = u1[1] * P1[2,:] - P1[1,:]
    
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3] 
    
    return X[:3]


def triangulate_with_best_projection(u0, u1, K, R1, T1, R2, T2):
    # Create projection matrices
    P1 = np.hstack((R1, T1))
    P2 = np.hstack((R2, T2))
    P3 = np.hstack((R2, -T2))
    P4 = np.hstack((R1, -T1))

    # Perform triangulation with each pair of projection matrices
    point1 = linear_ls_triangulation(u0, K @ P1, u1, K @ P2)
    point2 = linear_ls_triangulation(u0, K @ P1, u1, K @ P3)
    point3 = linear_ls_triangulation(u0, K @ P1, u1, K @ P4)
    point4 = linear_ls_triangulation(u0, K @ P2, u1, K @ P3)

    # Calculate reprojection errors for each case
    error1 = estimate_reprojection_error_for_camera([point1], K, R1, T1, [u0])
    error2 = estimate_reprojection_error_for_camera([point2], K, R1, -T1, [u0])
    error3 = estimate_reprojection_error_for_camera([point3], K, R2, T2, [u1])
    error4 = estimate_reprojection_error_for_camera([point4], K, R2, -T2, [u1])

    # Find the projection matrix that minimizes the reprojection error
    errors = [error1, error2, error3, error4]
    best_index = np.argmin(errors)
    
    # Return the best triangulated point and the corresponding projection matrix
    if best_index == 0:
        return point1, P1
    elif best_index == 1:
        return point2, P2
    elif best_index == 2:
        return point3, P3
    else:
        return point4, P4


def get_dominant_colors(image_path, k=5,n_init=10):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the image into a list of RGB values
    pixels = image.reshape((-1, 3))

    # Use k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(pixels)

    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors




def save_pcd_to_file(points_3d, colors, filename):
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the points and colors
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    # Save the PointCloud to a PCD file
    o3d.io.write_point_cloud(filename, pcd)



def visualize_3d_point_cloud(points, colors, title):

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points and colors
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors to [0, 1]

    # Create a Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)

    # Add the PointCloud to the visualizer
    vis.add_geometry(pcd)

    # Set the view control
    vis.get_render_option().point_size = 5
    vis.get_render_option().background_color = np.asarray([1, 1, 1])  # White background

    # Create a camera view
    front_view = vis.get_view_control()
    front_view.set_front([0, 0, -1])
    front_view.set_up([0, 1, 0])
    front_view.set_lookat([0, 0, 0])
    vis.run()

    
image_files = ['/home/upasana/Desktop/CV/rightcamera/Im_R_{}.png'.format(i) for i in range(1, 21)]
# image_files_2 = ['/home/upasana/Desktop/CV/leftcamera/Im_L_{}.png'.format(i) for i in range(1, 21)]
# image_files = image_files_1 + image_files_2
calibrate_and_undistort(image_files, (11, 7))

image_path1 = '/home/upasana/Desktop/CV/cereal_1.jpg'
image_path2 = '/home/upasana/Desktop/CV/cereal_2.jpg'


chessboard_size = (11, 7)
mtx, dist = calibrate_and_undistort(image_files, chessboard_size)
# print(mtx.shape)


calibration_matrix = np.load('/home/upasana/Desktop/CV/calibration_matrix.npy')
distortion_coeffs = np.load('/home/upasana/Desktop/CV/distortion_coeffs.npy')

# # Specify the image path
# image_path = '/path/to/your/image.jpg'

# # Undistort the image using pre-calibrated parameters
# undistorted_image = undistort_image(image_path, calibration_matrix, distortion_coeffs)




R1,R2,T,keypoints1,keypoints2, good_matches = find_correspondence_points(image_path1, image_path2, mtx, dist)

print("Number of Good Matches:", len(good_matches))


# print(F.shape)
T1=T
T2 = -T1

P0 = np.hstack((np.eye(3), np.zeros((3, 1))))
# P11, P12, P13, P14 = create_projection_matrices(mtx, R1, T)
P11 = np.matmul(mtx, np.hstack((R1, T1.reshape(3, 1))))
P12 = np.matmul(mtx, np.hstack((R1, T2.reshape(3, 1))))
P13 = np.matmul(mtx, np.hstack((R2, T1.reshape(3, 1))))
P14 = np.matmul(mtx, np.hstack((R2, T2.reshape(3, 1))))

print("Projection Matrix P0:")
print(P0)
print("\nProjection Matrices for Camera 1:")
print("P1:", P11)
print("P2:", P12)
print("P3:", P13)
print("P4:", P14)


# Iterate through corresponding points and triangulate
total_reprojection_error_cam1 = 0
total_reprojection_error_cam2 = 0

triangulated_points = []

# triangulated_points_cam1 = []
# triangulated_points_cam2 = []
triangulated_points = np.empty((0, 3))


# Iterate through corresponding points and triangulate
for i in range(len(good_matches)):
    match = good_matches[i]
    u0_homogeneous = np.append(keypoints1[match.queryIdx].pt, 1)
    u1_homogeneous = np.append(keypoints2[match.trainIdx].pt, 1)

    # print(f"u0_homogenous: {u0_homogeneous}")
    # print(f"u1_homogenous: {u1_homogeneous}")

    triangulated_point = linear_ls_triangulation(u0_homogeneous, P0, u1_homogeneous, P11)
    triangulated_points = np.append(triangulated_points, [triangulated_point], axis=0)

    #  # Perform triangulation with the original camera matrices
    # triangulated_point_cam1 = linear_ls_triangulation(u0_homogeneous, P0, u1_homogeneous, P12)
    # triangulated_point_cam2 = linear_ls_triangulation(u0_homogeneous, P0, u1_homogeneous, P12)

    # # Append the triangulated points to the lists
    # triangulated_points_cam1.append(triangulated_point_cam1)
    # triangulated_points_cam2.append(triangulated_point_cam2)


    # Triangulate the 3D point
    triangulated_point_homogeneous = np.append(triangulated_point,1)

    # Project the triangulated point back onto the image planes
    u0_reprojected_homogeneous = P0 @ triangulated_point_homogeneous
    u1_reprojected_homogeneous = P11@ triangulated_point_homogeneous

    # Normalize homogeneous coordinates
    u0_reprojected = u0_reprojected_homogeneous[:2] / u0_reprojected_homogeneous[2]
    u1_reprojected = u1_reprojected_homogeneous[:2] / u1_reprojected_homogeneous[2]


    # Calculate re-projection errors
    reprojection_error_cam1 = np.linalg.norm(u0_homogeneous[:2] - u0_reprojected)
    reprojection_error_cam2 = np.linalg.norm(u1_homogeneous[:2] - u1_reprojected)

    # Accumulate the errors
    total_reprojection_error_cam1 += reprojection_error_cam1
    total_reprojection_error_cam2 += reprojection_error_cam2

    triangulated_points = np.array(triangulated_points)


# triangulated_point = linear_ls_triangulation(u0_homogeneous, P0, u1_homogeneous, P12)
# print("Triangulated Point:", triangulated_point)
# print("Shape of triangulated_point:", triangulated_point.shape)

# # Final number of triangulated points for cameras 1 and 2
# num_triangulated_points_cam1 = len(triangulated_points_cam1)
# num_triangulated_points_cam2 = len(triangulated_points_cam2)

# # Print the final number of triangulated points
# print("Final Number of Triangulated Points for Camera 1:", num_triangulated_points_cam1)
# print("Final Number of Triangulated Points for Camera 2:", num_triangulated_points_cam2)


# Calculate mean re-projection errors
mean_reprojection_error_cam1 = total_reprojection_error_cam1 / len(good_matches)
mean_reprojection_error_cam2 = total_reprojection_error_cam2 / len(good_matches)

# Print the mean re-projection errors
print(f"Mean Re-projection Error for Camera 1: {mean_reprojection_error_cam1}")
print(f"Mean Re-projection Error for Camera 2: {mean_reprojection_error_cam2}")



dominant_colors1 = get_dominant_colors(image_path1, k=5, n_init=10)
dominant_colors2 = get_dominant_colors(image_path2, k=5, n_init=10)

print("Dominant Colors for Image 1:", dominant_colors1)
print("Dominant Colors for Image 2:", dominant_colors2)



print("Number of triangulated points:", triangulated_points.shape[0])

# save_pcd_to_file(triangulated_points_cam1, dominant_colors1, '/home/upasana/Desktop/CV/output_cam1.pcd')
save_pcd_to_file(triangulated_points, dominant_colors2, '/home/upasana/Desktop/CV/output_cam2.pcd')



# visualize_3d_point_cloud(triangulated_points_cam1, dominant_colors1, "Frontal_View_Cam1")
visualize_3d_point_cloud(triangulated_points, dominant_colors2, "Frontal_View_Cam2")




# # Save the combined PointCloud to a PCD file
# save_pcd_to_file(combined_points, combined_colors, '/home/upasana/Desktop/CV/output_combined.pcd')
# save_pcd_to_file(combined_points, combined_colors, '/home/upasana/Desktop/CV/output_1.pcd')

# visualize_3d_point_cloud(combined_points, combined_colors, "Frontal_View")
# # visualize_3d_point_cloud(triangulated_points_cam2, dominant_colors2, "Frontal_View_Cam2")
