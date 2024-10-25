import cv2
import numpy as np

max_depth_value = 500


def motion_estimation(image1_points, image2_points, intrinsic_matrix, depth, contours, max_depth=500):
    """
    Estimating motion of the left camera from previous left and current right imgaes 

    """  
    def reconstrucr_3d(image_points):
        points_3D = np.zeros((0, 3))
        outliers = []
        # Extract depth information to build 3D positions
        for indices, (u, v) in enumerate(image_points):
            z = depth[int(v), int(u)]

            # We will not consider depth greater than max_depth
            if z > max_depth:
                outliers.append(indices)
                continue

            # Using z we can find the x,y points in 3D coordinate using the formula
            x = z*(u-cx)/fx
            y = z*(v-cy)/fy

            # Stacking all the 3D (x,y,z) points
            points_3D = np.vstack([points_3D, np.array([x, y, z])])

        return points_3D, outliers
        
    image1_points = image1_points.squeeze(1)
    image2_points = image2_points.squeeze(1)

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    
    points_3D, outliers = reconstrucr_3d(image1_points)
    # Deleting the false depth points
    image1_points = np.delete(image1_points, outliers, 0)
    image2_points = np.delete(image2_points, outliers, 0)

    # Apply Ransac Algorithm to remove outliers
    _, rvec, translation_vector, _ = cv2.solvePnPRansac(
        points_3D, image2_points, intrinsic_matrix, None)

    rotation_matrix = cv2.Rodrigues(rvec)[0]


    if len(contours) > 0:
        contours = contours.squeeze(1)
        contours_3d, _ = reconstrucr_3d(contours)
        contour_3d = np.mean(contours_3d, axis=1)
    else:
        contour_3d = np.array([0,0,0])

    return rotation_matrix, translation_vector, contour_3d #, image1_points, image2_points

def find_contours(image, edited_image):
    # Step 2: Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    assert hsv_image is not None, "Failed to load image"
    # Step 3: Define the red color range
    # Lower and upper ranges for red color in HSV space
    lower_red_1 = np.array([0, 150, 100])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 150, 100])
    upper_red_2 = np.array([180, 255, 255])

    # Step 4: Create a mask for red color
    mask1 = cv2.inRange(hsv_image, lower_red_1, upper_red_1)  # Mask for lower red range
    mask2 = cv2.inRange(hsv_image, lower_red_2, upper_red_2)  # Mask for upper red range
    red_mask = mask1 + mask2  # Combine both masks

    # Step 5: Find contours on the mask
    # contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = image_difference(image, edited_image)
    points = []

    for index in range(len(contours)):
        # Approximate the contour to a polygon (optional: useful for boxes)
        epsilon = 0.02 * cv2.arcLength(contours[index], True)
        approx = cv2.approxPolyDP(contours[index], epsilon, True)
        #points = np.copy(approx)
        if index == 0:
            points = np.copy(approx)
        else:
            points = np.vstack((points, approx))

    return points

def image_difference(original_image, edited_image):
    # Convert both images to grayscale
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited_image, cv2.COLOR_BGR2GRAY)

    # Subtract the original image from the edited image
    difference = cv2.absdiff(edited_gray, original_gray)

    # Apply a threshold to highlight the drawn areas
    _, thresholded_difference = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection to detect edges of the drawing
    edges = cv2.Canny(thresholded_difference, 50, 150)

    # Find contours in the thresholded difference
    contours = cv2.findContours(thresholded_difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
