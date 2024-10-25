import cv2
import numpy as np
import os
import functions

class Camera:
    def __init__(self, cam: str):
        # Define intrinsic camera matrices and distortion coefficients for both cameras
        # Camera intrinsic parameters
        K_02 = np.array([[9.597910e+02, 0, 6.960217e+02],
                        [0, 9.569251e+02, 2.241806e+02],
                        [0, 0, 1]])
        D_02 = np.array([-3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02])
        R_02 = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                        [5.251945e-03, 9.999804e-01, -3.413835e-03],
                        [4.570332e-03, 3.389843e-03, 9.999838e-01]])
        T_02 = np.array([[5.956621e-02], [2.900141e-04], [2.577209e-03]])

        K_03 = np.array([[9.037596e+02, 0, 6.957519e+02],
                        [0, 9.019653e+02, 2.242509e+02],
                        [0, 0, 1]])
        D_03 = np.array([-3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02])
        R_03 = np.array([[9.995599e-01, 1.699522e-02, -2.431313e-02],
                        [-1.704422e-02, 9.998531e-01, -1.809756e-03],
                        [2.427880e-02, 2.223358e-03, 9.997028e-01]])
        T_03 = np.array([[-4.731050e-01], [5.551470e-03], [-5.250882e-03]])

        if cam == "left":
            K = K_02
            D = D_02
            Rc = R_02
            Tc = T_02

        elif cam == "right":
            K = K_03
            D = D_03
            Rc = R_03
            Tc = T_03

        else:
            raise ValueError("Error: only left or right as parameters.")
        
        self.K = K
        self.D = D
        self.Rc = Rc
        self.Tc = Tc

    def undistort_points(self, points):
        """ Undistort points using the camera's intrinsic parameters. """
        return cv2.undistortPoints(np.expand_dims(points, axis=1), self.K, self.D)

class StereoVisualOdometry:
    def __init__(self, camera_left, camera_right, R, T):
        self.camera_left = camera_left
        self.camera_right = camera_right
        # self.R = R
        # self.T = T
        # self.T2 = T + np.array((-self.get_baseline(), 0, 0))

    def get_baseline(self):
        return 0.54  # Baseline in meters for KITTI data

    # Function to rectify stereo images
    def rectify_images(self, left_image, right_image, K_left, K_right, D_left, D_right, width, height):
        R = np.eye(3)  # Assuming identity rotation for simplicity
        T = np.array([[-self.get_baseline(), 0.0, 0.0]]).reshape(3, 1)   # Translation in x-axis
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, (width, height), R, T)
        left_map1, left_map2 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, (width, height), cv2.CV_32FC1)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, (width, height), cv2.CV_32FC1)
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
        return left_rectified, right_rectified, Q

    # Function to compute disparity map
    def compute_disparity(self, left_rectified, right_rectified):
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=9,
                                    P1=8 * 3 * 9 ** 2, P2=32 * 3 * 9 ** 2, disp12MaxDiff=1,
                                    uniquenessRatio=15, speckleWindowSize=50, speckleRange=2)
        disparity = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0
        return disparity

    # Function to compute depth map
    def compute_depth(self, disparity, baseline, fx):
        with np.errstate(divide='ignore'):
            depth = (fx * baseline) / (disparity + 1e-5)
            depth[disparity <= 0] = 0  # Filter invalid disparity
        return depth
    
    def feature_maching(self, left_img1, left_img2):
        orb = cv2.ORB_create()
        # Previous frame keypoints and descriptors
        kp_prev_L, des_prev_L = orb.detectAndCompute(left_img1, None)
        kp_cur_L, des_cur_L = orb.detectAndCompute(left_img2, None)

        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match between previous left and previous right images
        matches_prev_cur_L = bf.match(des_prev_L, des_cur_L)

        # Extract matched points
        # Convert matched keypoints to lists of points for triangulation
        pts_prev_L = np.float32([kp_prev_L[m.queryIdx].pt for m in matches_prev_cur_L]).reshape(-1, 1, 2)
        pts_cur_L = np.float32([kp_cur_L[m.trainIdx].pt for m in matches_prev_cur_L]).reshape(-1, 1, 2)

        return pts_prev_L, pts_cur_L
    
    def process_frames(self, left_image_pre, right_image_pre, left_image_current, right_image_current):

        """ Process a pair of stereo images to estimate camera pose. """
        # Detect keypoints and extract descriptors

        height_pre, width_pre, _ = left_image_pre.shape
        height_current, width_current = left_image_current.shape
        
        # Rectify images
        left_rectified_pre, right_rectified_pre, Q = self.rectify_images(left_image_pre, right_image_pre, camera_left.K, 
                                                                         camera_right.K, camera_left.D, camera_right.D, width_pre, height_pre)
        
        # Look for red countours
        left_edited_rec = left_rectified_pre
        contours = functions.find_contours(left_rectified_pre, left_edited_rec)
        if len(contours) == 0:
            status = False
        else:
            status = True
        
        # convert color image to gray
        left_rectified_pre = cv2.cvtColor(left_rectified_pre, cv2.COLOR_BGR2GRAY)
        right_rectified_pre = cv2.cvtColor(right_rectified_pre, cv2.COLOR_BGR2GRAY)

        left_rectified_current, right_rectified_current, Q = self.rectify_images(left_image_current, right_image_current, camera_right.K, 
                                                                                 camera_right.K, camera_left.D, camera_right.D, width_current, height_current)

        # Compute disparity and depth
        disparity_map = self.compute_disparity(left_rectified_pre, right_rectified_pre)
        depth_map = self.compute_depth(disparity_map, self.get_baseline(), camera_left.K[0, 0])

        # Match features between the current left and next left image
        pts_prev_L, pts_cur_L = self.feature_maching(left_rectified_pre, left_rectified_current)

        # Calculate the rotation matrix and translation vector
        rotation_matrix, translation_vector, contour_3d = functions.motion_estimation(pts_prev_L, pts_cur_L, camera_left.K, depth_map, contours, max_depth=500)


        return rotation_matrix, translation_vector, contour_3d, status  # In case of failure


def sort_images(folder_path):
    # Check if the specified folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # List all files in the directory
    all_files = os.listdir(folder_path)

    # Filter for PNG and JPG files
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Sort the filtered files
    sorted_files = sorted(image_files)

    return sorted_files


# Example usage
if __name__ == "__main__":
    # Load stereo images (replace with actual paths)
    left_image_path = "kitti/2011_09_26_sync/2011_09_26_drive_0013_sync/image_02/data/"
    right_image_path = "kitti/2011_09_26_sync/2011_09_26_drive_0013_sync/image_03/data/"
    left_images = sort_images(left_image_path)
    right_images = sort_images(right_image_path)

    R = np.eye(3)  # Initial rotation matrix
    T = np.zeros((3, 1))  # Initial translation vector
    r = np.eye(3)
    t = np.zeros((3,1))

    # Create Camera objects for left and right cameras
    camera_left = Camera("left")
    camera_right = Camera("right")

    reference_coordinate = []

    for left_img0, right_img0, left_img1, right_img1 in zip(left_images[:len(left_images)-1], right_images[:len(right_images)-1], left_images[1:], right_images[1:]):
        
        print(".", end="")
        # read the images
        left_image_previous = cv2.imread(left_image_path + str(left_img0))
        right_image_previous = cv2.imread(right_image_path + str(right_img0))
        left_image_current = cv2.imread(left_image_path + str(left_img0), cv2.IMREAD_GRAYSCALE)
        right_image_current = cv2.imread(right_image_path + str(right_img0), cv2.IMREAD_GRAYSCALE)

        # Create a stereo visual odometry instance
        stereo_vo = StereoVisualOdometry(camera_left, camera_right, R, T)

        # Process the stereo images to get the rotation and translation
        R, T, contour_3d, status = stereo_vo.process_frames(left_image_previous, right_image_previous, left_image_current, right_image_current)

        if status:
            reference_coordinate.append([R, T, contour_3d])
        # print(T[2,0])
        r = r @ R
        t += T
        #print(t)

    if r is not None and t is not None:
        print("Rotation Matrix R:")
        print(r)
        print("Translation Vector T:")
        print(t)
        print("tracked coordinate:")
        print(reference_coordinate)
    else:
        print("Pose estimation failed.")
