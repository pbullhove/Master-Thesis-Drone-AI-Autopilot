#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image


import matplotlib.pyplot as plt
import numpy as np
import math
import json

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
drone_image_raw = None


# Constants
L1 = 79.0
L2 = 341.0
L3 = 863.0
L4 = 1110.0

RELATION_R_L1 = L4 / L1
RELATION_R_L2 = L4 / L2
RELATION_R_L3 = L4 / L3
RELATION_R_L4 = L4 / L4

# Image size
IMG_WIDTH = 640
IMG_HEIGHT = 360

# Relation between the traverse vector length and the longitudinal vector length
REL_TRAV_LONG = 0.23


######################
# Help functions #
######################
def make_median_blurry(image, blur_size):
    return cv2.medianBlur(image, blur_size)


def make_gaussian_blurry(image, blur_size):
    return cv2.GaussianBlur(image, (blur_size, blur_size), 0)


def make_circle_average_blurry(image, blur_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_size,blur_size))
    n_elements = np.float64(np.count_nonzero(kernel))
    kernel_norm = (kernel/n_elements)
    
    img_blurred = cv2.filter2D(image,-1,kernel_norm)

    return img_blurred


def hsv_save_image(image, label='image', is_gray=False):
    # folder = 'image_processing/detect_h/'
    folder = 'image_processing/cv_module/'
    if is_gray:
        cv2.imwrite(folder+label+".png", image)
    else:
        cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))
    return image


def load_hsv_image(filename):
    img = cv2.imread(filename) # import as BGR
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV
    hsv_save_image(hsv, "1_hsv") # Save image
    return hsv


def rgb_color_to_hsv(red, green, blue):
    bgr_color = np.uint8([[[blue,green,red]]])
    hsv_color = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
    return hsv_color[0][0].tolist()


def normalize_vector(vector):
    return vector / np.linalg.norm(vector) #, ord=1)


def hsv_to_opencv_hsv(hue, saturation, value):
    """ 
    Function that takes in hue, saturation and value in the ranges
        hue: [0, 360] degrees,  saturation: [0, 100] %,     value: [0, 100] %
    and converts it to OpenCV hsv which operates with the ranges
        hue: [0, 180],          saturation: [0, 255],       value: [0, 255]
    """
    converting_constant = np.array([0.5, 2.55, 2.55]) 
    return np.array([ hue, saturation, value])*converting_constant


def draw_dot(img, position, color):
    cX = position[1]
    cY = position[0]
    cv2.circle(img, (cX, cY), 3, color, -1)


def draw_arrow(img, start, end):
    return cv2.arrowedLine(img, (start[1], start[0]),
        (end[1], end[0]),
        color = (75,0,0), thickness = 1, tipLength = 0.4)


def calc_angle_between_vectors(vector_1, vector_2):

    v1_x = vector_1[0]
    v1_y = vector_1[1]

    v2_x = vector_2[0]
    v2_y = vector_2[1]

    angle = np.arctan2( v1_x*v2_y - v1_y*v2_x, v1_x*v2_x + v1_y*v2_y)

    # unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    # unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    # dot_product = np.dot(unit_vector_1, unit_vector_2)
    # angle = np.arccos(dot_product)
    return angle


def limit_point_to_be_inside_image(point):
    """ Make sure the point is inside the image 
        if it is not, move it to the closest border
    """
    smallest_x = 0
    smallest_y = 0
    largest_x = IMG_HEIGHT-1
    largest_y = IMG_WIDTH-1

    limited_point = np.int0(np.array([
        max(smallest_x, min(point[0], largest_x)),
        max(smallest_y, min(point[1], largest_y))
    ]))

    return limited_point


def print_header(text):
    text_length = len(text)
    border_line = "#"*(text_length+4)
    text_line = "# " + text + " #"

    print ""
    print border_line
    print text_line
    print border_line

# Colors to draw with
HSV_RED_COLOR = rgb_color_to_hsv(255,0,0)
HSV_BLUE_COLOR = rgb_color_to_hsv(0,0,255)
HSV_BLACK_COLOR = rgb_color_to_hsv(0,0,0)
HSV_YELLOW_COLOR = [30, 255, 255]
HSV_LIGHT_ORANGE_COLOR = [15, 255, 255]


##################
# Main functions #
##################
def get_white_mask(hsv):    
    lower_white = hsv_to_opencv_hsv(0, 0, 50)
    upper_white = hsv_to_opencv_hsv(360, 50, 100)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    _ , img_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    white_x, white_y = np.where(img_binary==255)
    if len(white_x) == 0: # No white visible
        return None
    else:
        return img_binary


def get_orange_mask(hsv):
    lower_orange = hsv_to_opencv_hsv(25, 50, 50)
    upper_orange = hsv_to_opencv_hsv(35, 100, 100)

    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange) 
    return orange_mask


def get_pixels_inside_orange(hsv):
    """ 
        Function that finds the orange in an image, make a bounding box around it,
        fits an ellipse in the bounding box
        and paints everything outside the ellipse in black.

        Returns the painted image and a boolean stating wheather any orange was found.
     """

    hsv_inside_orange = hsv.copy()

    hsv_orange_mask = get_orange_mask(hsv)  
    hsv_save_image(hsv_orange_mask, "2b_orange_mask", is_gray=True)

    orange_x, orange_y = np.where(hsv_orange_mask==255)
    
    if len(orange_x) == 0:
        # If no orange in image: return original image
        return hsv, False

    x_min = np.amin(orange_x)
    x_max = np.amax(orange_x)
    y_min = np.amin(orange_y)
    y_max = np.amax(orange_y)

    hsv_inside_orange[0:x_min,] = HSV_BLACK_COLOR
    hsv_inside_orange[x_max+1:,] = HSV_BLACK_COLOR

    hsv_inside_orange[:,0:y_min] = HSV_BLACK_COLOR
    hsv_inside_orange[:,y_max+1:] = HSV_BLACK_COLOR

    return hsv_inside_orange, True


def find_orange_arrow_point_old(hsv):
    # Parameters ###########################
    first_blur_size = 5
    second_blur_size = 49
    ignore_border_size = 3
    corner_harris_block_size = 4
    ignore_border_size = 3


    # Define valid intensity range for the median of a corner
    min_intensity_average = 10 # 20 # 170
    max_intensity_average = 85 # 100 # 240
    ########################################
    
    hsv_orange_mask = get_orange_mask(hsv)
    # hsv_save_image(hsv_orange_mask, "3_orange_only", is_gray=True)

    blur = make_gaussian_blurry(hsv_orange_mask, first_blur_size) 
    double_blur = make_gaussian_blurry(blur, second_blur_size)


    # Sub-pixel corner detection:
    dst = cv2.cornerHarris(blur,corner_harris_block_size,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(blur,np.float32(centroids),(5,5),(-1,-1),criteria)

    corner_x = np.int0(corners[:,1])
    corner_y = np.int0(corners[:,0])
    number_of_corners = len(corner_x)


    # Keep corners within the border limit
    x_min = np.array([ignore_border_size]*number_of_corners)
    x_max = np.array([IMG_HEIGHT - ignore_border_size]*number_of_corners)
    y_min = np.array([ignore_border_size]*number_of_corners)
    y_max = np.array([IMG_WIDTH - ignore_border_size]*number_of_corners)

    corners_clipped_on_border = corners[
        np.logical_and(
            np.logical_and(
                np.greater(corners[:,1], x_min),            # Add top limit
                np.less(corners[:,1], x_max)                # Add bottom limit
            ),
            np.logical_and(
                np.greater(corners[:,0], y_min),            # Add left limit
                np.less(corners[:,0], y_max)                # Add right limit
            )
        )
    ]

    corner_x = np.int0(corners_clipped_on_border[:,1])
    corner_y = np.int0(corners_clipped_on_border[:,0])
    
    if np.ndim(corner_x) == 0:
        number_of_corners = 1
    else:
        number_of_corners = len(corner_x)

    # Filter corner on median
    min_intensity = np.array([min_intensity_average]*number_of_corners)
    max_intensity = np.array([max_intensity_average]*number_of_corners)

    # Filter out the corners that belong to an "outer corner"
    # and outliers that are inside the white area
    corners_clipped_on_intensity = corners_clipped_on_border[
        np.logical_and(
            np.greater(
                double_blur[corner_x,corner_y],
                min_intensity
            ),                                                  # Add top limit
            np.less(
                double_blur[corner_x,corner_y],
                max_intensity
            )                                                   # Add bottom limit
        )
    ]

    corner_x = np.int0(corners_clipped_on_intensity[:,1])
    corner_y = np.int0(corners_clipped_on_intensity[:,0])


    # Draw corners
    img_all_corners = hsv.copy()
    img_all_corners[corner_x, corner_y] = HSV_RED_COLOR
    hsv_save_image(img_all_corners, "4_corners")

    if len(corners_clipped_on_intensity) == 1:
        return np.array([corner_x, corner_y])
    elif len(corners_clipped_on_intensity) == 0:
        print "Found zero corners"
    elif len(corners_clipped_on_intensity) > 1:
        print "Found too many corners"
    else:
        print "ERROR in find_orange_arrow_point()"
    
    return None


def find_white_centroid(hsv_white_only):
	# calculate moments of binary image
    M = cv2.moments(hsv_white_only)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Returned the transposed point,
    #  because of difference from OpenCV axis
    return np.array([cY, cX])


def calc_angle_centroid_arrowhead(centroid, arrowhead):
    """
        Calculates the angle between
        the line from the centroid to the arrowhead
        and the negative x-axis.
    """
    v_1 = arrowhead - centroid
    dx, dy = v_1[0], v_1[1]
    theta = np.degrees(np.arctan2(dy, -dx))
    return theta


def calc_angle_centroid_goal_point(centroid, goal_point):
    """
        Calculates the angle between
        the line from the centroid to the goal-point
        and the negative x-axis.
        The goal-point is between the two inner corners in the H.
    """
    v_2 = goal_point - centroid
    dx, dy = v_2[0], v_2[1]
    alpha = np.degrees(np.arctan2(-dy, -dx))
    return alpha


def alpha_to_theta(alpha):
    # Change alpha to be in range (-90, 270]
    # alpha -90 => 270
    # alpha -100 => 260
    if alpha < -90:
        alpha_sat = 360 + alpha
    else:
        alpha_sat = alpha

    theta = 90 - alpha_sat


def test_of_functions():
    centroid = np.array([0, 0])
    for quadrant in range(1,5):
        if quadrant == 1:
            ################
            # 1st quadrant #
            ################
            print "################"
            print "# 1st quadrant #"
            print "################"
            arrowhead = np.array([-2, 3])
            theta = calc_angle_centroid_arrowhead(centroid, arrowhead)
            print "Theta to arrowhead: ", theta

            goal_point = np.array([-3, -2])
            alpha = calc_angle_centroid_goal_point(centroid, goal_point)
            print "Alpha to goal_point: ", alpha

            # print "Theta (from alpha): ", alpha_to_theta(alpha)
            theta_from_alpha = alpha_to_theta(alpha)
            print

        elif quadrant == 2:
            ################
            # 2nd quadrant #
            ################
            print "################"
            print "# 2nd quadrant #"
            print "################"
            arrowhead = np.array([2, 3])
            theta = calc_angle_centroid_arrowhead(centroid, arrowhead)
            print "Theta to arrowhead: ", theta

            goal_point = np.array([-3, 2])
            alpha = calc_angle_centroid_goal_point(centroid, goal_point)
            print "Alpha to goal_point: ", alpha
            
            # print "Theta (from alpha): ", alpha_to_theta(alpha)
            theta_from_alpha = alpha_to_theta(alpha)
            print

        elif quadrant == 3:    
            ################
            # 3rd quadrant #
            ################

            print "################"
            print "# 3rd quadrant #"
            print "################"
            arrowhead = np.array([2, -3])
            theta = calc_angle_centroid_arrowhead(centroid, arrowhead)
            print "Theta to arrowhead: ", theta

            goal_point = np.array([3, 2])
            alpha = calc_angle_centroid_goal_point(centroid, goal_point)
            print "Alpha to goal_point: ", alpha 
            
            # print "Theta (from alpha): ", alpha_to_theta(alpha)
            theta_from_alpha = alpha_to_theta(alpha)
            print

        elif quadrant == 4: 
            ################
            # 4th quadrant #
            ################
            print "################"
            print "# 4th quadrant #"
            print "################"
            arrowhead = np.array([-2, -3])
            theta = calc_angle_centroid_arrowhead(centroid, arrowhead)
            print "Theta to arrowhead: ", theta

            goal_point = np.array([3, -2])
            alpha = calc_angle_centroid_goal_point(centroid, goal_point)
            print "Alpha to goal_point: ", alpha
            
            # print "Theta (from alpha): ", alpha_to_theta(alpha)
            theta_from_alpha = alpha_to_theta(alpha)
            print


# def detect_inner_corners(hsv_white_only, img_id = 0):
#     # Parameters ###########################
#     second_blur_size = 49
#     ignore_border_size = 3
#     corner_harris_block_size = 4

#     # Define valid intensity range for the median of a corner
#     min_intensity_average = 170
#     max_intensity_average = 240
#     ########################################

#     # hsv_origin = hsv.copy()
    
#     blur = make_gaussian_blurry(hsv_white_only, 5) 
#     double_blur = make_gaussian_blurry(blur, second_blur_size)
    
#     # Sub-pixel:
#     dst = cv2.cornerHarris(blur,corner_harris_block_size,3,0.04)
#     dst = cv2.dilate(dst,None)
#     ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
#     dst = np.uint8(dst)

#     # find centroids
#     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

#     # define the criteria to stop and refine the corners
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
#     corners = cv2.cornerSubPix(blur,np.float32(centroids),(5,5),(-1,-1),criteria)

#     # Filter the corners
#     number_of_corners = len(corners)
#     x_min = np.array([ignore_border_size]*number_of_corners)
#     x_max = np.array([IMG_HEIGHT - ignore_border_size]*number_of_corners)
#     y_min = np.array([ignore_border_size]*number_of_corners)
#     y_max = np.array([IMG_WIDTH - ignore_border_size]*number_of_corners)

#     # Keep corners within the border limit
#     corners_clipped_on_border = corners[
#         np.logical_and(
#             np.logical_and(
#                 np.greater(corners[:,1], x_min),            # Add top limit
#                 np.less(corners[:,1], x_max)                # Add bottom limit
#             ),
#             np.logical_and(
#                 np.greater(corners[:,0], y_min),            # Add left limit
#                 np.less(corners[:,0], y_max)                # Add right limit
#             )
#         )
#     ]
#     corner_x = np.int0(corners_clipped_on_border[:,1])
#     corner_y = np.int0(corners_clipped_on_border[:,0])
#     number_of_corners = len(corner_x)

#     min_intensity = np.array([min_intensity_average]*number_of_corners)
#     max_intensity = np.array([max_intensity_average]*number_of_corners)

#     # Filter out the corners that belong to an "outer corner"
#     # and outliers that are inside the white area
#     corners_clipped_on_intensity = corners_clipped_on_border[
#         np.logical_and(
#             np.greater(
#                 double_blur[corner_x,corner_y],
#                 min_intensity
#             ),                                                  # Add top limit
#             np.less(
#                 double_blur[corner_x,corner_y],
#                 max_intensity
#             )                                                   # Add bottom limit
#         )
#     ]
#     corner_x = np.int0(corners_clipped_on_intensity[:,1])
#     corner_y = np.int0(corners_clipped_on_intensity[:,0])

#     if np.ndim(corner_x) == 0:
#         number_of_corners = 1
#         corners = np.array([[corner_x, corner_y]])
#     else:
#         corners = np.stack((corner_x, corner_y), axis=1)
#         number_of_corners = len(corners)

#     if number_of_corners == 0 or number_of_corners > 4:
#         print("Invalid number of corners")
#         return None, None


#     ######################
#     # Define the corners #

#     if number_of_corners == 1:
#         corner_0 = corners[0]

#     if number_of_corners == 2:
#         left_corner_id = np.argmin(corner_y)
#         right_corner_id = 1 - left_corner_id

#         corner_0 = corners[left_corner_id]
#         corner_1 = corners[right_corner_id]


#     if number_of_corners == 3:
#         distances = np.array([
#             np.linalg.norm(corners[0] - corners[1]),
#             np.linalg.norm(corners[1] - corners[2]),
#             np.linalg.norm(corners[2] - corners[0])
#         ])

#         min_dist_id = np.argmin(distances)

#         if min_dist_id == 0:
#             relevant_corners = np.stack((corners[0], corners[1]), axis=0)
#         elif min_dist_id == 1:
#             relevant_corners = np.stack((corners[1], corners[2]), axis=0)
#         elif min_dist_id == 2:
#             relevant_corners = np.stack((corners[2], corners[0]), axis=0)
#         else:
#             print("ERROR: In fn. detect_sub_pixecl_corners(); min_dist_id out of bounds")
#             return None, None

#         dist_0_1, dist_1_2, dist_2_0 = distances[0], distances[1], distances[2]

#         # Pick the corner with the lowest y-coordinate
#         left_corner_id = np.argmin(relevant_corners, axis=0)[1]
#         right_corner_id = 1 - left_corner_id

#         corner_0 = relevant_corners[left_corner_id]
#         corner_1 = relevant_corners[right_corner_id]


#     if number_of_corners == 4:
#         # For the first corner, chose the corner closest to the top
#         # This will belong to the top cross-bar
#         top_corner_id = np.argmin(corner_x)
#         top_corner = corners[top_corner_id]
#         top_corner_stack = np.array([top_corner]*3)
#         rest_corners = np.delete(corners, top_corner_id, 0)
#         dist = np.linalg.norm(rest_corners - top_corner_stack, axis=1)

#         # For the second corner, chose the corner closest to top corner
#         top_corner_closest_id = np.argmin(dist)
#         top_corner_closest = rest_corners[top_corner_closest_id]

#         relevant_corners = np.stack((top_corner, top_corner_closest), axis=0)

#         # Choose the corner with the lowest y-coordinate as the first corner
#         left_corner_id = np.argmin(relevant_corners, axis=0)[1]
#         right_corner_id = 1 - left_corner_id

#         corner_0 = relevant_corners[left_corner_id]
#         corner_1 = relevant_corners[right_corner_id]


#     ##################################
#     # Define goal point and direction #'
#     goal_point = None
#     dir_vector = None

#     # Calculate spatial gradient
#     dy, dx	= cv2.spatialGradient(blur) # Invert axis, since OpenCV operates with x=column, y=row
    
#     if number_of_corners == 1:
#         grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
#         grad_0_normalized = normalize_vector(grad_0)

#         goal_point_offset = 5 # px-distance
#         goal_point = np.int0(corner_0 + grad_0_normalized*goal_point_offset)

#         dir_vector = grad_0_normalized

#     else:
#         grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
#         grad_1 = np.array([dx[corner_1[0]][corner_1[1]], dy[corner_1[0]][corner_1[1]]])
#         grad_sum = grad_0 + grad_1
#         grad_sum_normalized = normalize_vector(grad_sum)

#         cross_over_vector = corner_1 - corner_0
#         goal_point = np.int0(corner_0 + 0.5*(cross_over_vector))
#         goal_value = double_blur[goal_point[0]][goal_point[1]]


#     if number_of_corners == 2 and goal_value < 200:
#         print "The two corners form the longitudinal bar"
#         # Choose to focus on the left point

#         v_long = corner_0 - corner_1
#         v_long_norm = normalize_vector(v_long)

#         grad_0_normalized = normalize_vector(grad_0)

#         if grad_0_normalized[0] < 0:
#             long_bar_norm = np.array([-cross_over_vector[1], cross_over_vector[0]])
#         else:
#             long_bar_norm = np.array([cross_over_vector[1], -cross_over_vector[0]])
#         long_bar_norm_normalized = normalize_vector(long_bar_norm)
        

#         dist_to_top_edge = corner_0[0]
#         dist_to_bottom_edge = IMG_HEIGHT - corner_0[0]
#         dist_to_nearest_edge = min(dist_to_top_edge, dist_to_bottom_edge)
#         goal_point_offset = dist_to_nearest_edge / 2 # px-distance
#         goal_point = np.int0(corner_0 + long_bar_norm_normalized*goal_point_offset)

#         dir_vector = v_long_norm
    
#     elif number_of_corners != 1:
#         if grad_sum_normalized[0] < 0:
#             cross_over_norm = np.array([-cross_over_vector[1], cross_over_vector[0]])
#         else:
#             cross_over_norm = np.array([cross_over_vector[1], -cross_over_vector[0]])

#         dir_vector = normalize_vector(cross_over_norm)


#     ###################
#     # Draw the result #

#     direction_length = 20
#     direction_point = np.int0(goal_point + dir_vector*direction_length)

#     # img_grad = hsv_origin.copy()
#     # img_grad = cv2.arrowedLine(img_grad, (goal_point[1], goal_point[0]),
#     #     (direction_point[1], direction_point[0]),
#     #     color = (255,0,0), thickness = 2, tipLength = 0.5)

#     # hsv_save_image(img_grad, "5_gradient")
#     # hsv_save_image(img_grad, str(img_id) +"_gradient")


#     return goal_point, corners


def find_harris_corners(img, block_size):
    """ Using sub-pixel method from OpenCV """
    dst = cv2.cornerHarris(img, block_size, 3, 0.04)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Flip axis
    corners[:,[0, 1]] = corners[:,[1, 0]]

    return corners


def clip_corners_on_border(corners, border_size):
    # Filter the corners
    number_of_corners = len(corners)
    x_min = np.array([border_size]*number_of_corners)
    x_max = np.array([IMG_HEIGHT - border_size]*number_of_corners)
    y_min = np.array([border_size]*number_of_corners)
    y_max = np.array([IMG_WIDTH - border_size]*number_of_corners)

    corner_x = np.int0(corners[:,0])
    corner_y = np.int0(corners[:,1])

    # Keep corners within the border limit
    corners_clipped_on_border = corners[
        np.logical_and(
            np.logical_and(
                np.greater(corner_x, x_min),            # Add top limit
                np.less(corner_x, x_max)                # Add bottom limit
            ),
            np.logical_and(
                np.greater(corner_y, y_min),            # Add left limit
                np.less(corner_y, y_max)                # Add right limit
            )
        )
    ]
    corner_x = np.int0(corners_clipped_on_border[:,0])
    corner_y = np.int0(corners_clipped_on_border[:,1])
    
    if np.ndim(corner_x) == 0:
        number_of_corners = 1
        corners = np.array([[corner_x, corner_y]])
    else:
        corners = np.stack((corner_x, corner_y), axis=1)
        number_of_corners = len(corners)
    if number_of_corners == 0:
        return None
    else:
        return corners


def clip_corners_on_intensity(corners, img, average_filter_size):
    """
        Filter out the corners that belong to a right-angled corner
        i.e. corners with a mean intensity value around 255/4~64   number_of_corners = len(corners)
    """
    value_per_degree = 255.0/360.0
    min_degree, max_degree = 60, 120 # +- 30 from 90 degrees

    # Since 255 is white and 0 is black, subtract from 255
    # to get black intensity instead of white intensity
    min_average_intensity = 255 - max_degree*value_per_degree
    max_average_intensity = 255 - min_degree*value_per_degree

    number_of_corners = len(corners)
    print number_of_corners

    min_intensity = np.array([min_average_intensity]*number_of_corners)
    max_intensity = np.array([max_average_intensity]*number_of_corners)

    img_average_intensity = make_circle_average_blurry(img, average_filter_size)

    corner_x = np.int0(corners[:,0])
    corner_y = np.int0(corners[:,1])

    corners_clipped_on_intensity = corners[
        np.logical_and(
            np.greater(
                img_average_intensity[corner_x,corner_y],
                min_intensity
            ),                                                  # Add top limit
            np.less(
                img_average_intensity[corner_x,corner_y],
                max_intensity
            )                                                   # Add bottom limit
        )
    ]
    corner_x = np.int0(corners_clipped_on_intensity[:,0])
    corner_y = np.int0(corners_clipped_on_intensity[:,1])
    
    if np.ndim(corner_x) == 0:
        corners = np.array([[corner_x, corner_y]])
        intensities = np.array([img_average_intensity[corner_x, corner_y]])
        number_of_corners = 1
    else:
        corners = np.stack((corner_x, corner_y), axis=1)
        intensities = np.array(img_average_intensity[corner_x, corner_y])
        number_of_corners = len(corners)
    print number_of_corners

    print "intensities: ", intensities

    if number_of_corners == 0:
        return None, None
    else:
        return corners, intensities


def find_right_angled_corners(img):
    # Parameters ###########################
    average_filter_size = 19 # 19
    ignore_border_size = 3
    corner_harris_block_size = 4

    # Define valid intensity range for the median of a corner
    min_intensity_average = 170
    max_intensity_average = 240
    ########################################

    corners = find_harris_corners(img, corner_harris_block_size)
    corners = clip_corners_on_border(corners, ignore_border_size)
    if corners is None:
        print "Found no corners"
        return None

    corners, intensities = clip_corners_on_intensity(corners, img, average_filter_size)
    if corners is None:
        print "Found no corners"
        return None, None

    return corners, intensities


def get_gradient_of_point(point, dx, dy):
    point_x = point[0]
    point_y = point[1]
    gradient = np.array([dx[point_x][point_y], dy[point_x][point_y]])
    return gradient


def find_goal_point(hsv_white_only, inner_corners):
    number_of_corners = len(inner_corners)
    average_filter_size = 19
    img_average_intensity = make_circle_average_blurry(hsv_white_only, average_filter_size)

    ######################
    # Define the inner_corners #

    if number_of_corners == 1:
        corner_0 = inner_corners[0]

    elif number_of_corners == 2:
        left_corner_id = np.argmin(inner_corners[:,1])
        right_corner_id = 1 - left_corner_id

        corner_0 = inner_corners[left_corner_id]
        corner_1 = inner_corners[right_corner_id]
        
    elif number_of_corners == 3:
        distances = np.array([
            np.linalg.norm(inner_corners[0] - inner_corners[1]),
            np.linalg.norm(inner_corners[1] - inner_corners[2]),
            np.linalg.norm(inner_corners[2] - inner_corners[0])
        ])

        median = np.median(distances)
        median_id = np.where(distances == median)[0][0]
        print "median:", median
        print "median_id:", median_id

        if median_id == 0:
            relevant_corners = np.stack((inner_corners[0], inner_corners[1]), axis=0)
        elif median_id == 1:
            relevant_corners = np.stack((inner_corners[1], inner_corners[2]), axis=0)
        elif median_id == 2:
            relevant_corners = np.stack((inner_corners[2], inner_corners[0]), axis=0)
        else:
            print("ERROR: In fn. detect_sub_pixecl_corners(); min_dist_id out of bounds")
            return None, None

        dist_0_1, dist_1_2, dist_2_0 = distances[0], distances[1], distances[2]

        # Pick the corner with the lowest y-coordinate
        left_corner_id = np.argmin(relevant_corners, axis=0)[1]
        right_corner_id = 1 - left_corner_id

        corner_0 = relevant_corners[left_corner_id]
        corner_1 = relevant_corners[right_corner_id]

    elif number_of_corners == 4:
        # For the first corner, chose the corner closest to the top
        # This will belong to the top cross-bar
        top_corner_id = np.argmin(inner_corners[:,0]) # Find lowest x-index
        top_corner = inner_corners[top_corner_id]
        top_corner_stack = np.array([top_corner]*3)
        rest_corners = np.delete(inner_corners, top_corner_id, 0)
        dist = np.linalg.norm(rest_corners - top_corner_stack, axis=1)

        # For the second corner, chose the corner closest to top corner
        top_corner_closest_id = np.argmin(dist)
        top_corner_closest = rest_corners[top_corner_closest_id]

        relevant_corners = np.stack((top_corner, top_corner_closest), axis=0)

        # Choose the corner with the lowest y-coordinate as the first corner
        left_corner_id = np.argmin(relevant_corners, axis=0)[1]
        right_corner_id = 1 - left_corner_id

        corner_0 = relevant_corners[left_corner_id]
        corner_1 = relevant_corners[right_corner_id]

    else:
        print("Invalid number of corners")
        return None, None

    ##################################
    # Define goal point and direction #'
    goal_point = None
    goal_direction = None

    # Calculate spatial gradient
    dy, dx	= cv2.spatialGradient(hsv_white_only) # Invert axis, since OpenCV operates with x=column, y=row
    
    if number_of_corners == 1:
        grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
        grad_0_normalized = normalize_vector(grad_0)

        goal_point_offset = 5 # px-distance
        goal_point = np.int0(corner_0 + grad_0_normalized*goal_point_offset)

        goal_direction = grad_0_normalized
    else:
        grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
        grad_1 = np.array([dx[corner_1[0]][corner_1[1]], dy[corner_1[0]][corner_1[1]]])
        grad_sum = grad_0 + grad_1
        grad_sum_normalized = normalize_vector(grad_sum)

        cross_over_vector = corner_1 - corner_0
        goal_point = np.int0(corner_0 + 0.5*(cross_over_vector))
        goal_value = img_average_intensity[goal_point[0]][goal_point[1]]


    if (number_of_corners == 2 or number_of_corners == 3) and goal_value < 200:
        print "The two inner_corners form the longitudinal bar"
        # wp
        # Choose to focus on the uppermost corner,
        # since it is assumed the orientation is approximately right
        longitudinal_corners = np.stack((corner_0, corner_1))

        upper_corner_id = np.argmin(longitudinal_corners[:,0])
        lower_corner_id = 1 - upper_corner_id

        upper_corner = longitudinal_corners[upper_corner_id]
        lower_corner = longitudinal_corners[lower_corner_id]

        longitudinal_vector = upper_corner - lower_corner
        longitudinal_unit_vector = normalize_vector(longitudinal_vector)

        upper_corner_gradient = get_gradient_of_point(upper_corner, dx, dy)
        upper_corner_unit_gradient = normalize_vector(upper_corner_gradient)


        longitudinal_length = np.linalg.norm(longitudinal_vector)
        travese_length = longitudinal_length*REL_TRAV_LONG
        length_from_upper_corner_to_goal = travese_length/2.0

        arrow_length = 10
        grad_start = upper_corner
        grad_end = np.int0(upper_corner + upper_corner_unit_gradient*arrow_length)

        angle = calc_angle_between_vectors(longitudinal_vector, upper_corner_unit_gradient)
        l_x, l_y = longitudinal_unit_vector[0], longitudinal_unit_vector[1]
        if angle > 0:
            # This means the gradient is pointing to the left
            dir_vector = np.array([-l_y, l_x])
        else:
            # The gradient is pointing to the right (or is parallel)
            dir_vector = np.array([l_y, -l_x])
        goal_point = np.int0(upper_corner + dir_vector*length_from_upper_corner_to_goal)
        goal_point = limit_point_to_be_inside_image(goal_point)
        
        print "Angle:", np.degrees(angle)

        hsv_drawn_vectors = hsv_white_only.copy()
        hsv_drawn_vectors = draw_arrow(hsv_drawn_vectors, grad_start, grad_end)
        hsv_drawn_vectors = draw_arrow(hsv_drawn_vectors, upper_corner, goal_point)
        hsv_save_image(hsv_drawn_vectors, "0_drawn_vectors", is_gray=True)

        goal_direction = longitudinal_unit_vector
    
    elif number_of_corners != 1:
        if grad_sum_normalized[0] < 0:
            cross_over_norm = np.array([-cross_over_vector[1], cross_over_vector[0]])
        else:
            cross_over_norm = np.array([cross_over_vector[1], -cross_over_vector[0]])

        goal_direction = normalize_vector(cross_over_norm)


    ###################
    # Draw the result #

    # direction_length = 20
    # direction_point = np.int0(goal_point + dir_vector*direction_length)

    # img_grad = hsv_origin.copy()
    # img_grad = cv2.arrowedLine(img_grad, (goal_point[1], goal_point[0]),
    #     (direction_point[1], direction_point[0]),
    #     color = (255,0,0), thickness = 2, tipLength = 0.5)

    # hsv_save_image(img_grad, "5_gradient")
    # hsv_save_image(img_grad, str(img_id) +"_gradient")


    return goal_point, goal_direction


def find_orange_arrow_point(hsv):
    hsv_orange_mask = get_orange_mask(hsv)
    hsv_orange_mask = make_gaussian_blurry(hsv_orange_mask, 5) 

    hsv_save_image(hsv_orange_mask, "0_orange_mask", is_gray=True)
    
    hsv_orange_mask_inverted = cv2.bitwise_not(hsv_orange_mask)
    hsv_save_image(hsv_orange_mask_inverted, "0_orange_mask_inverted", is_gray=True)

    orange_corners, intensities = find_right_angled_corners(hsv_orange_mask_inverted)

    if orange_corners is None:
        return None

    number_of_corners_found = len(orange_corners)

    value_per_degree = 255.0/360.0
    ideal_angle = 90
    ideal_intensity = 255-ideal_angle*value_per_degree
    ideal_intensities = np.array([ideal_intensity]*number_of_corners_found)

    diff_intensities = np.absolute(np.array(ideal_intensities-intensities))
    print "Ideal intensities:", ideal_intensities
    print "Found intensities:", intensities
    
    print "Diff intensities:", diff_intensities

    print orange_corners

    hsv_orange_corners = hsv.copy()
    hsv_orange_corners_filtered = hsv.copy()
    for corner in orange_corners:
        draw_dot(hsv_orange_corners, corner, HSV_RED_COLOR)
    hsv_save_image(hsv_orange_corners, "0_orange_corners_before")

    if number_of_corners_found == 1:
        return orange_corners[0]
    elif number_of_corners_found > 1:
        print "Too many orange corners found, choose the best"
        best_corner_id = np.argmin(diff_intensities)
        best_corner = orange_corners[best_corner_id]

        draw_dot(hsv_orange_corners_filtered, best_corner, HSV_RED_COLOR)
        hsv_save_image(hsv_orange_corners_filtered, "0_orange_corners_after")

        return best_corner
    else:
        print "No corners found"
        return None


def calc_dist_to_landing_platform(centroid, arrowhead):
    print
    print "Dist to landing platform calculations"
    

    object_length_px = np.linalg.norm(centroid - arrowhead)
    print "Object length: ", object_length_px

    object_length_m = 0.288
    focal_length = 374.67
    distance_to_landing_platform = (object_length_m*focal_length) / object_length_px

    d = 750
    f = 374.67
    s_r = 288
    s_o = 133.09
    h_i = 360

    h_s = (f*s_r*h_i)/(s_o*d)

    print "Sensor height:", h_s

    # print "Pixel size:", object_length_px
    # lens_angle = 92.0
    # deg_per_px = lens_angle/np.sqrt(IMG_HEIGHT**2 + IMG_WIDTH**2)
    
    # angle = object_length_px * deg_per_px
    # if angle <= 0 or angle >=90:
    #     print "Invalid angle (", angle, ")"
    #     return None

    # angle_radians = np.radians(angle)

    # distance_to_landing_platform = object_length_m*1/np.tan(angle_radians)

    print distance_to_landing_platform
    print
    return distance_to_landing_platform


def calculate_position(center_px, radius_px):
    real_radius = 375 # mm (750mm in diameter / 2)
    # real_radius = 288 # previous value

    # Center of image
    x_0 = IMG_HEIGHT/2.0
    y_0 = IMG_WIDTH/2.0

    # Find distances from center of image to center of LP
    d_x = x_0 - center_px[0]
    d_y = y_0 - center_px[1]

    est_z = real_radius*focal_length / radius_px # - 59.4
    
    # Camera is placed 150 mm along x-axis of the drone
    # Since the camera is pointing down, the x and y axis of the drone
    # is the inverse of the x and y axis of the camera
    est_x = -(est_z * d_x / focal_length) - 150 
    est_y = -(est_z * d_y / focal_length)

    return np.array([est_x, est_y, est_z])


def run(img_count = 0):
    suffix = "_flip_horizontal"
    suffix = ""
    filepath = "dataset/low_flight_dataset_02/image_"+str(img_count)+suffix+".png"
    hsv = load_hsv_image(filepath)

    img_marked = hsv.copy()
    img_corners = hsv.copy()

    hsv_inside_orange, isOrangeVisible = get_pixels_inside_orange(hsv)
    hsv_save_image(hsv_inside_orange, "2_inside_orange")


    hsv_white_only = get_white_mask(hsv_inside_orange)
    if hsv_white_only is None:
        hsv_white_only = get_white_mask(hsv)
    hsv_white_only = make_gaussian_blurry(hsv_white_only, 5)    

    hsv_save_image(hsv_white_only, "3_white_only", is_gray=True)

    centroid = find_white_centroid(hsv_white_only)
    print "centroid:", centroid


    if isOrangeVisible:
        arrowhead = find_orange_arrow_point(hsv) # for testing with an angle : + np.array([1, 1])
        print "arrowhead:", arrowhead

        if arrowhead is None: # Arrowhead is not visible or too many corners found
            print "Arrowhead is not found"
            return None, None
        else:
            found_length_px = np.linalg.norm(centroid-arrowhead)
            print "length:", found_length_px
            return found_length_px, centroid

            theta = calc_angle_centroid_arrowhead(centroid, arrowhead)
            # print "Theta to arrowhead: ", theta
            draw_dot(img_marked, arrowhead, HSV_BLUE_COLOR)


            calc_dist_to_landing_platform(centroid, arrowhead)

    inner_corners, intensities = find_right_angled_corners(hsv_white_only)

    for corner in inner_corners:
        draw_dot(img_corners, corner, HSV_LIGHT_ORANGE_COLOR)
        draw_dot(img_marked, corner, HSV_LIGHT_ORANGE_COLOR)
    hsv_save_image(img_corners, "0_corners")

    goal, goal_direction = find_goal_point(hsv_white_only, inner_corners)
    draw_dot(img_marked, goal, HSV_YELLOW_COLOR)

    goal_direction_length = 50
    goal_direction_end = np.int0(goal + goal_direction*goal_direction_length)
    draw_arrow(img_marked, goal, goal_direction_end)

    # heading, inner_corners = detect_inner_corners(hsv_white_only)
    # if inner_corners is None:
    #     print "No inner corners found"
    # else:
    #     for corner in inner_corners:
    #         draw_dot(img_marked, corner, HSV_LIGHT_ORANGE_COLOR)

    #     draw_dot(img_marked, heading, HSV_YELLOW_COLOR)
    
    draw_dot(img_marked, centroid, HSV_RED_COLOR)


    hsv_save_image(img_marked, "3_marked")


# Examples
#
# In dataset_01 (20 images)
# 18 for two traverse corners
#
#
# In dataset_02 (42 images)
# 3 for three corners, where the two traversal is pointing down
# 29 arrowhead not visible (should be)
# 1 three corners showing


single_image_index = 4
single_image = False

# json_filepath = "dataset/low_flight_dataset_02/low_flight_dataset.json"
# with open(json_filepath) as json_file:
#     data = json.load(json_file)
#     index = str(single_image_index)
#     z = data[index]['ground_truth'][2]
#     print "GT_z:", z

results_gt = []
results_est = []

total_results = []

focal_length = 374.67

if single_image:
    print "###########################"
    print "# Now testing on image", str(single_image_index).rjust(2), "#"
    print "###########################"
    run(single_image_index)
else:
    for i in range(3):
        # answer = raw_input("Press enter for next image")
        print ""
        print "###########################"
        print "# Running CV module on image", i, "#"
        print "###########################"
        length, centroid = run(i)
        print ""
        print "# Preprocessing"
        print "length, centroid:", length, centroid
        print ""

        if length is not None:
            json_filepath = "dataset/low_flight_dataset_02/low_flight_dataset.json"
            with open(json_filepath) as json_file:
                data = json.load(json_file)
                index = str(i)
                gt_x = data[index]['ground_truth'][0]*1000
                gt_y = data[index]['ground_truth'][1]*1000
                gt_z = data[index]['ground_truth'][2]*1000
                results_gt.append([gt_x, gt_y, gt_z])
                print "GT: ", gt_x, gt_y, gt_z

            print calculate_position(centroid, length)

            x_0 = IMG_HEIGHT/2.0
            y_0 = IMG_WIDTH/2.0
            d_x = x_0 - centroid[0]
            d_y = y_0 - centroid[1]

            est_z = 288*focal_length / length - 59.4
            
            # Camera is placed 150 mm along x-axis of the drone
            est_x = -(est_z * d_x / focal_length) - 150 
            est_y = -(est_z * d_y / focal_length)

            print "Est:", est_x, est_y, est_z
            results_est.append([est_x, est_y, est_z])
            total_results.append([gt_z, est_z])


np_results_gt = np.array(results_gt)
np_results_est = np.array(results_est)
# print "Results ground truth"
# print np_results_gt
# print "Results estimate"
# print np_results_est

print_header("Showing results")

n_results = len(np_results_gt)
print "n_results:", n_results
rjust = 7
print "||  gt_x   |  est_x  ||  gt_y   |  est_y  ||  gt_z   |  est_z  ||"
print "-----------------------------------------------------------------"
for i in range(n_results):

    text_gt_x = '{:.2f}'.format(round(np_results_gt[i][0], 2)).rjust(rjust)
    text_est_x = '{:.2f}'.format(round(np_results_est[i][0], 2)).rjust(rjust)

    text_gt_y = '{:.2f}'.format(round(np_results_gt[i][1], 2)).rjust(rjust)
    text_est_y = '{:.2f}'.format(round(np_results_est[i][1], 2)).rjust(rjust)

    text_gt_z = '{:.2f}'.format(round(np_results_gt[i][2], 2)).rjust(rjust)
    text_est_z = '{:.2f}'.format(round(np_results_est[i][2], 2)).rjust(rjust)

    print "||", text_gt_x, "|",text_est_x, \
        "||", text_gt_y, "|",text_est_y, \
        "||", text_gt_z, "|",text_est_z, "||"

# print total_results

# print ""
# print "Diffs:"
# diffs = []
# diffs_sum = 0
# for res in total_results:
#     diff = res[0]-res[1]
#     diffs_sum += diff
#     # print diff
#     diffs.append(res[0]-res[1])
# n_diffs = len(total_results)
# print n_diffs
# avr = diffs_sum / n_diffs
# print avr
# run()