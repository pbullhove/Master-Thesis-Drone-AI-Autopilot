#!/usr/bin/env python

"""
This "single" version incorporates all estimates from the three different methods into one.

"""

import rospy
from std_msgs.msg import Empty, Int8
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

import numpy as np
import math

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

import color_settings
import config as cfg

# For ground truth callback:
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

# Settings
global_is_simulator = cfg.is_simulator

save_images = cfg.save_images
draw_on_images = cfg.draw_on_images
use_test_image = cfg.use_test_image


if global_is_simulator:
    camera_offset_x = 150 # mm
    camera_offset_z = -45 # mm
else:
    camera_offset_x = -60 # mm
    camera_offset_z = -45 # mm

# Constants
D_H_SHORT = 4.0 # cm
D_H_LONG = 12.0 # cm
D_ARROW = 30.0 # cm
D_RADIUS = 39.0 # cm

IMG_WIDTH = 640
IMG_HEIGHT = 360

global_image = None

#################
# ROS functions #
#################
def image_callback(data):
    global global_image

    try:
        global_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
    except CvBridgeError as e:
        rospy.loginfo(e)


global_ground_truth = None
def gt_callback(data):
    global global_ground_truth
    global_ground_truth = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


##################
# Help functions #
##################
def make_gaussian_blurry(image, blur_size):
    return cv2.GaussianBlur(image, (blur_size, blur_size), 0)


def make_circle_average_blurry(image, blur_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_size,blur_size))

    n_elements = np.float64(np.count_nonzero(kernel))
    kernel_norm = (kernel/n_elements)

    img_blurred = cv2.filter2D(image,-1,kernel_norm)
    return img_blurred


def hsv_save_image(image, label='image', is_gray=False):
    if image is None:
        print "The image with label " + label + " is none"
    folder = './image_processing/'
    if save_images:
        if is_gray:
            cv2.imwrite(folder+label+".png", image)
        else:
            cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))
    return image


def load_bgr_image(filename):
    bgr = cv2.imread(filename) # import as BGR
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert to HSV
    return bgr


def rgb_color_to_hsv(red, green, blue):
    bgr_color = np.uint8([[[blue,green,red]]])
    hsv_color = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
    return hsv_color[0][0].tolist()


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def hsv_to_opencv_hsv(hue, saturation, value):
    """
    Function that takes in hue, saturation and value in the ranges
        hue: [0, 360] degrees,  saturation: [0, 100] %,     value: [0, 100] %
    and converts it to OpenCV hsv which operates with the ranges
        hue: [0, 180],          saturation: [0, 255],       value: [0, 255]
    """
    converting_constant = np.array([0.5, 2.55, 2.55])
    return np.array([ hue, saturation, value])*converting_constant


def draw_dot(img, position, color, size=3):
    if draw_on_images:
        cX = np.int0(position[1])
        cY = np.int0(position[0])
        cv2.circle(img, (cX, cY), size, color, -1)


def draw_arrow(img, start, end):
    if draw_on_images:
        return cv2.arrowedLine(img,
            (np.int0(start[1]), np.int0(start[0])),
            (np.int0(end[1]), np.int0(end[0])),
            color = (75,0,0), thickness = 1, tipLength = 0.4)


def calc_angle_between_vectors(vector_1, vector_2):
    v1_x = vector_1[0]
    v1_y = vector_1[1]

    v2_x = vector_2[0]
    v2_y = vector_2[1]

    return np.arctan2( v1_x*v2_y - v1_y*v2_x, v1_x*v2_x + v1_y*v2_y)


def limit_point(point):
    limits_min = [0,0]
    limits_max = [IMG_HEIGHT-1, IMG_WIDTH-1]
    clipped = np.int0(np.clip(point, limits_min, limits_max))
    return clipped


def get_normal_vector(bw_white_mask, corner_a, corner_b, is_short_side):
    hsv_normals = bw_white_mask.copy()

    vector_between_a_b = corner_a - corner_b

    vector_length = np.linalg.norm(vector_between_a_b)
    unit_vector_between_a_b = normalize_vector(vector_between_a_b)

    v_x, v_y = vector_between_a_b
    normal_unit_vector_left = normalize_vector(np.array([-v_y, v_x]))
    normal_unit_vector_right = normalize_vector(np.array([v_y, -v_x]))

    if is_short_side:
        check_length = vector_length / 3.0
        sign = 1 # Go outwards from the corners
    else:
        short_length = vector_length * D_H_SHORT/D_H_LONG

        check_length = short_length / 3.0
        sign = -1 # Go inwards from the corners

    check_left_a = limit_point(corner_a + \
        sign*unit_vector_between_a_b*check_length + \
        normal_unit_vector_left*check_length)
    check_left_b = limit_point(corner_b - \
        sign*unit_vector_between_a_b*check_length + \
        normal_unit_vector_left*check_length)

    check_right_a = limit_point(corner_a + \
        sign*unit_vector_between_a_b*check_length + \
        normal_unit_vector_right*check_length)
    check_right_b = limit_point(corner_b - \
        sign*unit_vector_between_a_b*check_length + \
        normal_unit_vector_right*check_length)

    value_left_a = bw_white_mask[check_left_a[0]][check_left_a[1]]
    value_left_b = bw_white_mask[check_left_b[0]][check_left_b[1]]
    value_right_a = bw_white_mask[check_right_a[0]][check_right_a[1]]
    value_right_b = bw_white_mask[check_right_b[0]][check_right_b[1]]

    avr_left = value_left_a/2.0 + value_left_b/2.0
    avr_right = value_right_a/2.0 + value_right_b/2.0

    if avr_left > avr_right:
        return normal_unit_vector_left
    else:
        return normal_unit_vector_right


def is_mask_touching_border(bw_mask, padding = 0):
    """
    The padding is added when finding the white centroid.
    The reason is that there is a border of black around the image,
    that origins from the get_pixels_inside_green() method.
    """

    top_border =    bw_mask[padding,:]
    bottom_border = bw_mask[IMG_HEIGHT-1-padding,:]
    left_border =   bw_mask[:,padding]
    right_border =  bw_mask[:,IMG_WIDTH-1-padding]

    sum_border = np.sum(top_border) + \
        np.sum(bottom_border) + \
        np.sum(left_border) + \
        np.sum(right_border)

    if sum_border != 0:
        # Then the mask is toughing the border
        return True
    else:
        return False


# Colors to draw with
HSV_RED_COLOR = rgb_color_to_hsv(255,0,0)
HSV_BLUE_COLOR = rgb_color_to_hsv(0,0,255)
HSV_BLACK_COLOR = rgb_color_to_hsv(0,0,0)

HSV_YELLOW_COLOR = [30, 255, 255]
HSV_LIGHT_ORANGE_COLOR = [15, 255, 255]


if global_is_simulator:
    HUE_LOW_WHITE, HUE_HIGH_WHITE, SAT_LOW_WHITE, SAT_HIGH_WHITE, VAL_LOW_WHITE, VAL_HIGH_WHITE, \
    HUE_LOW_ORANGE, HUE_HIGH_ORANGE, SAT_LOW_ORANGE, SAT_HIGH_ORANGE, VAL_LOW_ORANGE, VAL_HIGH_ORANGE, \
    HUE_LOW_GREEN, HUE_HIGH_GREEN, SAT_LOW_GREEN, SAT_HIGH_GREEN, VAL_LOW_GREEN, VAL_HIGH_GREEN = color_settings.SIMULATOR_COLOR_LIMITS
else:
    HUE_LOW_WHITE, HUE_HIGH_WHITE, SAT_LOW_WHITE, SAT_HIGH_WHITE, VAL_LOW_WHITE, VAL_HIGH_WHITE, \
    HUE_LOW_ORANGE, HUE_HIGH_ORANGE, SAT_LOW_ORANGE, SAT_HIGH_ORANGE, VAL_LOW_ORANGE, VAL_HIGH_ORANGE, \
    HUE_LOW_GREEN, HUE_HIGH_GREEN, SAT_LOW_GREEN, SAT_HIGH_GREEN, VAL_LOW_GREEN, VAL_HIGH_GREEN = color_settings.REAL_COLOR_LIMITS


##################
# Main functions #
##################
def get_mask(hsv, hue_low, hue_high, sat_low, sat_high, val_low, val_high):
    lower_color = hsv_to_opencv_hsv(hue_low, sat_low, val_low)
    upper_color = hsv_to_opencv_hsv(hue_high, sat_high, val_high)

    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask_x, mask_y = np.where(mask==255)
    if len(mask_x) == 0: # No color visible
        return None
    else:
        return mask


def get_white_mask(hsv):
    return get_mask(hsv, HUE_LOW_WHITE, HUE_HIGH_WHITE, SAT_LOW_WHITE, SAT_HIGH_WHITE, VAL_LOW_WHITE, VAL_HIGH_WHITE)


def get_orange_mask(hsv):
    return get_mask(hsv, HUE_LOW_ORANGE, HUE_HIGH_ORANGE, SAT_LOW_ORANGE, SAT_HIGH_ORANGE, VAL_LOW_ORANGE, VAL_HIGH_ORANGE)


def get_green_mask(hsv):
    return get_mask(hsv, HUE_LOW_GREEN, HUE_HIGH_GREEN, SAT_LOW_GREEN, SAT_HIGH_GREEN, VAL_LOW_GREEN, VAL_HIGH_GREEN)


def flood_fill(img, start=(0,0)):
    h,w = img.shape
    seed = start

    mask = np.zeros((h+2,w+2),np.uint8) # Adding a padding of 1

    if img[start[1]][start[0]] == 255: # If the starting point is already filled, return empty mask
        mask = mask[1:h+1,1:w+1] # Removing the padding
        return mask

    floodflags = 8
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    num,img,mask,rect = cv2.floodFill(img, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
    mask = mask[1:h+1,1:w+1] # Removing the padding

    return mask


def get_pixels_inside_green(hsv):
    """
        Function that finds the green in an image
        and paints everything else black.

        Returns the painted image.
     """
    bw_green_mask = get_green_mask(hsv)
    if bw_green_mask is None:
        return hsv

    kernel = np.ones((15,15),np.uint8)

    bw_closed_mask = cv2.dilate(bw_green_mask, kernel, iterations = 6)
    # bw_closed_mask = cv2.erode(bw_closed_mask, kernel, iterations = 6, borderType=cv2.BORDER_REFLECT_101)
    # bw_closed_mask = cv2.erode(bw_closed_mask, kernel, iterations = 6) # borderType=cv2.BORDER_WRAP)
    bw_closed_mask = cv2.erode(bw_closed_mask, kernel, iterations = 6, borderType=cv2.BORDER_CONSTANT, borderValue = 0)

    hsv_ellipse = hsv.copy()
    hsv_ellipse[bw_closed_mask==0] = HSV_BLACK_COLOR

    return hsv_ellipse


def get_h_area(hsv):
    # For every side, find either all pixels green or the furtherst orange pixel
    green_mask = get_green_mask(hsv)
    orange_mask = get_orange_mask(hsv)
    white_mask = get_white_mask(hsv)

    # hsv_save_image(green_mask, "green_mask", is_gray=True)
    # hsv_save_image(orange_mask, "orange_mask", is_gray=True)
    # hsv_save_image(white_mask, "white_mask", is_gray=True)

    # If no orange in image,
    # green and white on the edges,
    # paint all black.
    x_min = IMG_HEIGHT-1
    x_max = 0
    y_min = IMG_WIDTH - 1
    y_max = 0

    # Check orange area
    if orange_mask is not None:
        orange_x, orange_y = np.where(orange_mask==255)
        if len(orange_x) != 0:
            x_min = np.amin(orange_x)
            x_max = np.amax(orange_x)
            y_min = np.amin(orange_y)
            y_max = np.amax(orange_y)

    # Check green and white area
    margin = 10
    top_bottom_margin = IMG_WIDTH - margin
    left_right_margin = IMG_HEIGHT - margin

    if (green_mask is not None) and (white_mask is not None):
        # Top
        if np.sum(np.logical_or(green_mask[0,:] > 0, white_mask[0,:] > 0)) > top_bottom_margin:
            # All on top is green
            x_min = 0

        # Right
        if np.sum(np.logical_or(green_mask[:,IMG_WIDTH-1] > 0, white_mask[:,IMG_WIDTH-1] > 0)) > left_right_margin:
            # All on right is green
            y_max = IMG_WIDTH-1

        # Bottom
        if np.sum(np.logical_or(green_mask[IMG_HEIGHT-1,:] > 0, white_mask[IMG_HEIGHT-1,:] > 0)) > top_bottom_margin:
            # All on bottom is green
            x_max = IMG_HEIGHT-1

        # Left
        if np.sum(np.logical_or(green_mask[:,0] > 0, white_mask[:,0] > 0)) > left_right_margin:
            # All on left is green
            y_min = 0


    hsv_inside_orange = hsv.copy()

    hsv_inside_orange[0:x_min,] = HSV_BLACK_COLOR
    hsv_inside_orange[x_max+1:,] = HSV_BLACK_COLOR
    hsv_inside_orange[:,0:y_min] = HSV_BLACK_COLOR
    hsv_inside_orange[:,y_max+1:] = HSV_BLACK_COLOR

    return hsv_inside_orange


def find_white_centroid(hsv):
    bw_white_mask = get_white_mask(hsv)
    if bw_white_mask is None:
        return None
    hsv_save_image(bw_white_mask, "3_white_mask", is_gray=True)

    if is_mask_touching_border(bw_white_mask, padding=42):
        return None

    bw_white_mask = make_gaussian_blurry(bw_white_mask, 5)

    M = cv2.moments(bw_white_mask) # calculate moments of binary image

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # Returnes the transposed point,
    #  because of difference from OpenCV axis
    return np.array([cY, cX])


def find_harris_corners(img):
    """ Using sub-pixel method from OpenCV
    Inspiration: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """

    block_size = 7      # It is the size of neighbourhood considered for corner detection
    aperture_param = 9  # Aperture parameter of Sobel derivative used.
    k_param = 0.04      # Harris detector free parameter in the equation. range: [0.04, 0.06]

    bw_blurred = make_gaussian_blurry(img, 9)

    dst = cv2.cornerHarris(bw_blurred, block_size, aperture_param, k_param)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)
    corners = cv2.cornerSubPix(bw_blurred, np.float32(centroids), (5,5), (-1,-1), criteria)

    # Flip axis
    corners[:,[0, 1]] = corners[:,[1, 0]]

    # Remove the first corner, as this always becomes the center (the reason for this is currently unknown)
    return corners[1:]


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
                np.less(corner_x, x_max)),              # Add bottom limit
            np.logical_and(
                np.greater(corner_y, y_min),            # Add left limit
                np.less(corner_y, y_max))               # Add right limit
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
        i.e. corners with a mean intensity value around 255/4~64
    """
    value_per_degree = 255.0/360.0
    min_degree, max_degree = 60, 120 # +- 30 from 90 degrees
    # min_degree, max_degree = 0, 2000

    # Since 255 is white and 0 is black, subtract from 255
    # to get black intensity instead of white intensity
    # min_average_intensity = 255 - max_degree*value_per_degree
    # max_average_intensity = 255 - min_degree*value_per_degree

    min_average_intensity = 200
    max_average_intensity = 255


    number_of_corners = len(corners)
    min_intensity = np.array([min_average_intensity]*number_of_corners)
    max_intensity = np.array([max_average_intensity]*number_of_corners)

    img_average_intensity = make_circle_average_blurry(img, average_filter_size)

    corner_x = np.int0(corners[:,0])
    corner_y = np.int0(corners[:,1])

    corners_clipped_on_intensity = corners[
        np.logical_and(
            np.greater(                                                 # Add top limit
                img_average_intensity[corner_x,corner_y],
                min_intensity),
            np.less(                                                    # Add bottom limit
                img_average_intensity[corner_x,corner_y],
                max_intensity)
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

    if number_of_corners == 0:
        return None, None
    else:
        return corners, intensities


def clip_corners_not_right(corners, img, average_filter_size):
    gaussian_blur_size = 51
    sigmaX = 0
    # 191 is the mathemathical theoretical value for a right angled corner
    value_limit_low = 160
    value_limit_high = 210

    bw_blurred = img.copy()
    bw_blurred = cv2.GaussianBlur(bw_blurred, (gaussian_blur_size,gaussian_blur_size), sigmaX)

    bgr = cv2.cvtColor(bw_blurred, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    marked = hsv.copy()

    # Text settings
    font = cv2.FONT_HERSHEY_DUPLEX
    fontScale = 0.8
    color = (150, 0, 0)
    thickness = 1

    for corner in corners:
        corner_x = np.int0(corner[0])
        corner_y = np.int0(corner[1])
        value = bw_blurred[corner_x, corner_y]
        # Using cv2.putText() method
        image = cv2.putText(bw_blurred, str(value), (corner_y-25, corner_x+40), font,
                    fontScale, color, thickness, cv2.LINE_AA)

    values = bw_blurred[corners[:,0], corners[:,1]]

    is_to_keep = np.logical_and(value_limit_low < values, values < value_limit_high)
    new_corners = corners[is_to_keep]
    new_values = values[is_to_keep]

    # cv2.circle(bw_blurred, (320,180), np.int0(gaussian_blur_size/2.0), (0,0,0), 1)

    color = HSV_YELLOW_COLOR
    fontScale = 0.4
    for corner in corners:
        corner_x = np.int0(corner[0])
        corner_y = np.int0(corner[1])
        value = bw_blurred[corner_x, corner_y]
        image = cv2.putText(marked, str(value), (corner_y, corner_x), font,
                    fontScale, color, thickness, cv2.LINE_AA)
        marked[corner_x, corner_y] = HSV_RED_COLOR

    hsv_save_image(marked, "marked")

    hsv_save_image(bw_blurred, "blurred", is_gray=True)

    return new_corners, new_values



def find_right_angled_corners(img):
    # Parameters ###########################
    ignore_border_size = 7 # 7, 20
    average_filter_size = 19
    check_angle_filter_size = 19
    ########################################

    corners = find_harris_corners(img)

    corners = clip_corners_on_border(corners, ignore_border_size)
    if corners is None:
        return None, None

    corners, intensities = clip_corners_not_right(corners, img, check_angle_filter_size)
    if corners is None:
        return None, None

    return corners, intensities


def find_orange_arrowhead(hsv):
    bw_orange_mask = get_orange_mask(hsv)
    if bw_orange_mask is None:
        return None

    # bw_orange_mask = make_gaussian_blurry(bw_orange_mask, 9)
    bw_orange_mask_inverted = cv2.bitwise_not(bw_orange_mask)

    hsv_save_image(bw_orange_mask_inverted, "3_orange_mask_inverted", is_gray=True)

    orange_corners, intensities = find_right_angled_corners(bw_orange_mask_inverted)
    if orange_corners is None:
        return None

    number_of_corners_found = len(orange_corners)

    ideal_intensity = 191 # 75% of 255
    ideal_intensities = np.array([ideal_intensity]*number_of_corners_found)

    diff_intensities = np.absolute(np.array(ideal_intensities-intensities))

    if number_of_corners_found == 1:
        return orange_corners[0]
    elif number_of_corners_found > 1:
        # Too many orange corners found, choose the best
        best_corner_id = np.argmin(diff_intensities)
        best_corner = orange_corners[best_corner_id]
        return best_corner
    else:
        # No corners found
        return None


def calculate_position(center_px, radius_px):
    focal_length = 374.67 if cfg.is_simulator else 720.0 # but acutally 687.0
    real_radius = 390 # mm (780mm in diameter / 2)

    # Center of image
    x_0 = IMG_HEIGHT/2.0
    y_0 = IMG_WIDTH/2.0

    # Find distances from center of image to center of LP
    d_x = x_0 - center_px[0]
    d_y = y_0 - center_px[1]

    est_z = real_radius*focal_length / radius_px

    # Camera is placed 150 mm along x-axis of the drone
    # Since the camera is pointing down, the x and y axis of the drone
    # is the inverse of the x and y axis of the camera
    est_x = -((est_z * d_x / focal_length) + camera_offset_x) # mm adjustment for translated camera frame in x direction
    est_y = -(est_z * d_y / focal_length)

    est_z += camera_offset_z # mm adjustment for translated camera frame in z direction

    position = np.array([est_x, est_y, est_z]) / 1000.0

    return position


def fit_ellipse(points):
    x = points[1]
    y = IMG_HEIGHT-points[0]

    D11 = np.square(x)
    D12 = x*y
    D13 = np.square(y)
    D1 = np.array([D11, D12, D13]).T
    D2 = np.array([x, y, np.ones(x.shape[0])]).T

    S1 = np.dot(D1.T,D1)
    S2 = np.dot(D1.T,D2)
    S3 = np.dot(D2.T,D2)

    try:
        inv_S3 = np.linalg.inv(S3)
    except np.linalg.LinAlgError:
        print("fit_ellipse(): Got singular matrix")
        return None

    T = - np.dot(inv_S3, S2.T) # for getting a2 from a1

    M_inner = S1 + np.dot(S2, T)

    C1 = np.array([
        [0, 0, 0.5],
        [0, -1, 0],
        [0.5, 0, 0]
    ])

    M = np.dot(C1, M_inner) # This premultiplication can possibly be made more efficient

    eigenvalues, eigenvectors = np.linalg.eig(M)
    cond = 4*eigenvectors[0]*eigenvectors[2] - np.square(eigenvectors[0])
    a1 = eigenvectors[:,cond > 0]

    # Choose the first if there are two eigenvectors with cond > 0
    # NB! I am not sure if this is always correct
    if a1.shape[1] > 1:
        a1 = np.array([a1[:,0]]).T

    if a1.shape != (3,1): # Make sure a1 has content
        print("fit_ellipse(): a1 not OK")
        return None

    a = np.concatenate((a1, np.dot(T, a1)))[:,0] # Choose the inner column with [:,0]

    if np.any(np.iscomplex(a)):
        print("Found complex number")
        return None
    else:
        return a


def get_ellipse_parameters(green_ellipse):
    edges = cv2.Canny(green_ellipse,100,200)
    # edges = cv2.Canny(image=image, first_threshold=100, second_threshold=200, apertureSize = 3)
    result = np.where(edges == 255)

    ellipse = fit_ellipse(result)

    if ellipse is None:
        return None

    A = ellipse[0]
    B = ellipse[1]
    C = ellipse[2]
    D = ellipse[3]
    E = ellipse[4]
    F = ellipse[5]

    if B**2 - 4*A*C >= 0:
        print("get_ellipse_parameters(): Shape found is not an ellipse")
        return None

    inner_square = math.sqrt( (A-C)**2 + B**2)
    outside = 1.0 / (B**2 - 4*A*C)
    a = outside * math.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( (A+C) + inner_square))
    b = outside * math.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( (A+C) - inner_square))

    x_raw = (2.0*C*D - B*E) / (B*B - 4.0*A*C)
    y_raw = (2.0*A*E - B*D) / (B*B - 4.0*A*C)

    x_0 = (IMG_HEIGHT - 1) - y_raw
    y_0 = x_raw

    ellipse_and_a_b = np.array([x_0,y_0,a,b])
    return ellipse_and_a_b


def evaluate_ellipse(hsv):
    """ Use the green ellipse to find:
        center, radius, angle
    """
    bw_green_mask = get_green_mask(hsv)
    if bw_green_mask is None:
        return None, None, None

    if is_mask_touching_border(bw_green_mask):
        return None, None, None

    bw_green_ellipse = flood_fill(bw_green_mask, start=(0,0))
    hsv_save_image(bw_green_ellipse, "3_green_ellipse", is_gray=True)

    ellipse_parameters = get_ellipse_parameters(bw_green_ellipse)
    if ellipse_parameters is None:
        return None, None, None

    # Choose the largest of the a and b parameter for the radius
    # Choose angle = 0 since it is not possible to estimate from the ellipse
    center_px = ellipse_parameters[0:2]
    radius_px = np.amax(np.abs(ellipse_parameters[2:4]))
    angle = 0 # No angle estimate avaiable with this method

    hsv_canvas_ellipse = hsv.copy()
    draw_dot(hsv_canvas_ellipse, center_px, HSV_BLUE_COLOR)
    hsv_save_image(hsv_canvas_ellipse, "4_canvas_ellipse")

    return center_px, radius_px, angle


def est_rotation(center, Arrow):
    # if center == [None, None]:
    #     return None
    # arrow_vector = np.array(np.array(est_center_of_bb(Arrow)) - np.array(center))
    # arrow_unit_vector = normalize_vector(arrow_vector)
    # arrow_unit_vector_yx = np.array([arrow_unit_vector[1], arrow_unit_vector[0]])
    # rad = calc_angle_between_vectors(arrow_unit_vector_yx, np.array([0,1]))
    # deg = rad2deg(rad)
    # print("center: ", center)
    # print("arrow: ", Arrow)

    dy = center[0] - Arrow[0]
    dx = Arrow[1] - center[1]
    rads = math.atan2(dy,dx)
    # degs = rads*180 / math.pi
    rads *= -1
    return rads



def evaluate_arrow(hsv, hsv_inside_green):
    """ Use the arrow to find:
        center, radius, angle
    """
    center_px = find_white_centroid(hsv_inside_green)
    arrowhead_px = find_orange_arrowhead(hsv)


    if (center_px is not None) and (arrowhead_px is not None):

        arrow_vector = np.array(arrowhead_px - center_px)
        arrow_unit_vector = normalize_vector(arrow_vector)
        ref_vector = np.array([0,1])



        # angle = calc_angle_between_vectors(arrow_vector, ref_vector)
        angle = est_rotation(center_px, arrowhead_px)
        arrow_length_px = np.linalg.norm(arrow_vector)
        # Use known relation between the real radius and the real arrow length
        # to find the radius length in pixels

        radius_length_px = arrow_length_px * D_RADIUS / D_ARROW

        hsv_canvas_arrow = hsv.copy()
        draw_dot(hsv_canvas_arrow, center_px, HSV_RED_COLOR)
        # draw_dot(hsv_canvas_arrow, arrowhead_px, HSV_RED_COLOR)
        draw_dot(global_hsv_canvas_all, arrowhead_px, HSV_RED_COLOR)

        hsv_save_image(hsv_canvas_arrow, "4_canvas_arrow")

        return center_px, radius_length_px, angle

    else:
        return None, None, None


def get_relevant_corners(inner_corners):
    n_inner_corners = len(inner_corners)

    # For the first corner, chose the corner closest to the top
    # This will belong to the top cross-bar
    top_corner_id = np.argmin(inner_corners[:,0]) # Find lowest x-index
    top_corner = inner_corners[top_corner_id]
    top_corner_stack = np.array([top_corner]*(n_inner_corners-1))
    rest_corners = np.delete(inner_corners, top_corner_id, 0)
    dist = np.linalg.norm(rest_corners - top_corner_stack, axis=1)

    # For the second corner, chose the corner closest to top corner
    top_corner_closest_id = np.argmin(dist)
    top_corner_closest = rest_corners[top_corner_closest_id]

    relevant_corners = np.stack((top_corner, top_corner_closest), axis=0)

    return relevant_corners


def evaluate_inner_corners(hsv):
    """ Use the inner corners to find:
        center, radius, angle
    """
    hsv_canvas = hsv.copy()

    bw_white_mask = get_white_mask(hsv)
    if bw_white_mask is None:
        return None, None, None
    # bw_white_mask = make_gaussian_blurry(bw_white_mask, 9) #5

    hsv_save_image(bw_white_mask, "2_white_only", is_gray=True)

    inner_corners, _ = find_right_angled_corners(bw_white_mask)

    bw_white_mask = make_gaussian_blurry(bw_white_mask, 9) #5
    average_filter_size = 19
    img_average_intensity = make_circle_average_blurry(bw_white_mask, average_filter_size)

    if (inner_corners is not None):
        n_inner_corners = len(inner_corners)
        if (n_inner_corners > 1) and (n_inner_corners <= 5):
            unique_corners = np.vstack({tuple(row) for row in inner_corners}) # Removes duplicate corners. Deprecated way of doing this, but works for now.

            for corner in unique_corners:
                draw_dot(hsv_canvas, corner, HSV_YELLOW_COLOR)

            corner_a, corner_b = get_relevant_corners(unique_corners)
            c_m = (corner_a + corner_b)/2.0 # Finds the mid point between the corners
            c_m_value = img_average_intensity[np.int0(c_m[0])][np.int0(c_m[1])]

            # draw_dot(hsv_canvas, corner_a, HSV_RED_COLOR)
            # draw_dot(hsv_canvas, corner_b, HSV_LIGHT_ORANGE_COLOR)
            draw_dot(global_hsv_canvas_all, corner_a, HSV_YELLOW_COLOR)
            draw_dot(global_hsv_canvas_all, corner_b, HSV_YELLOW_COLOR)

            draw_dot(hsv_canvas, c_m, HSV_BLUE_COLOR)
            hsv_save_image(hsv_canvas, "3_canvas")

            if c_m_value > 191: # The points are on a short side
                is_short_side = True
                normal_vector = get_normal_vector(bw_white_mask, corner_a, corner_b, is_short_side)
                normal_unit_vector = normalize_vector(normal_vector)

                length_short_side = np.linalg.norm(corner_a - corner_b)
                length_long_side = length_short_side * D_H_LONG/D_H_SHORT
                length_to_center = - length_long_side / 2.0
                length_radius = length_short_side * D_RADIUS/D_H_SHORT

                forward_unit_vector = normal_unit_vector

            else: # The points are on a long side
                is_short_side = False
                normal_vector = get_normal_vector(bw_white_mask, corner_a, corner_b, is_short_side)
                normal_unit_vector = normalize_vector(normal_vector)

                length_long_side = np.linalg.norm(corner_a - corner_b)
                length_short_side = length_long_side * D_H_SHORT/D_H_LONG
                length_to_center = length_short_side / 2.0
                length_radius = length_long_side * D_RADIUS/D_H_LONG

                forward_unit_vector = normalize_vector(corner_a - corner_b)

            end = c_m + forward_unit_vector*10
            # draw_arrow(hsv_canvas, c_m, end)
            draw_arrow(global_hsv_canvas_all, c_m, end)

            center = c_m + normal_unit_vector*length_to_center
            # draw_dot(hsv_canvas, center, HSV_BLUE_COLOR)

            # hsv_save_image(hsv_canvas, "3_canvas")

            neg_x_axis = np.array([-1,0])
            angle = calc_angle_between_vectors(forward_unit_vector, neg_x_axis)
            hsv_canvas_inner_corners = hsv.copy()
            draw_dot(hsv_canvas_inner_corners, center, HSV_LIGHT_ORANGE_COLOR)
            hsv_save_image(hsv_canvas_inner_corners, "4_canvas_inner_corners")

            return center, length_radius, angle

    return None, None, None


def get_estimate(hsv, count, current_ground_truth):
    global pub_est_ellipse
    global pub_est_arrow
    global pub_est_corners

    global pub_est_error_ellipse
    global pub_est_error_arrow
    global pub_est_error_corners

    global global_hsv_canvas_all
    method_of_choice = None # Updated to show which method is used for each timestep

    hsv_save_image(hsv, '0_hsv')
    global_hsv_canvas_all = hsv.copy()

    hsv_inside_green = get_pixels_inside_green(hsv)
    hsv_save_image(hsv_inside_green, '0_hsv_inside_green')

    hsv_h_area = get_h_area(hsv)

    white_mask = get_white_mask(hsv)
    if white_mask is not None:
        hsv_save_image(white_mask, '1_white_mask', is_gray=True)

    orange_mask = get_orange_mask(hsv)
    if orange_mask is not None:
        hsv_save_image(orange_mask, '1_orange_mask', is_gray=True)

    green_mask = get_green_mask(hsv)
    green_toughing_edge = False
    if green_mask is not None:
        hsv_save_image(green_mask, '1_green_mask', is_gray=True)

        if (green_mask[0,:] > 0).any() or \
                (green_mask[:,IMG_WIDTH-1] > 0).any() or \
                (green_mask[IMG_HEIGHT-1,:] > 0).any() or \
                (green_mask[:,0] > 0).any():
            green_toughing_edge = True

    msg = Twist()

    center_px_from_ellipse, radius_length_px_from_ellipse, angle_from_ellipse = evaluate_ellipse(hsv)
    center_px_from_arrow, radius_length_px_from_arrow, angle_from_arrow = evaluate_arrow(hsv, hsv_inside_green) # use hsv or hsv_inside_green
    center_px_from_inner_corners, radius_px_length_from_inner_corners, angle_from_inner_corners = evaluate_inner_corners(hsv_h_area)

    ellipse_available = (center_px_from_ellipse is not None) and (radius_length_px_from_ellipse != 0)
    arrow_available = (center_px_from_arrow is not None) and (radius_length_px_from_arrow != 0)
    corners_available = (center_px_from_inner_corners is not None) and (radius_px_length_from_inner_corners != 0)

    ############
    # Method 1 #
    ############
    if ellipse_available:
        # method_of_choice = "ellipse"
        center_px, radius_length_px, angle_rad = center_px_from_ellipse, radius_length_px_from_ellipse, angle_from_ellipse
        est_ellipse_x, est_ellipse_y, est_ellipse_z = calculate_position(center_px, radius_length_px)
        est_ellipse_angle = np.degrees(angle_rad)

        msg.linear.x = est_ellipse_x
        msg.linear.y = est_ellipse_y
        msg.linear.z = est_ellipse_z
        msg.angular.z = est_ellipse_angle
        pub_est_ellipse.publish(msg)

        try:
            msg.linear.x = est_ellipse_x - current_ground_truth[0]
            msg.linear.y = est_ellipse_y - current_ground_truth[1]
            msg.linear.z = est_ellipse_z - current_ground_truth[2]
            msg.angular.z = est_ellipse_angle - current_ground_truth[5]
            pub_est_error_ellipse.publish(msg)
        except TypeError as e:
            pass
        draw_dot(global_hsv_canvas_all, center_px, HSV_BLUE_COLOR, size=10)
    else:
        msg.linear.x = np.nan
        msg.linear.y = np.nan
        msg.linear.z = np.nan
        msg.angular.z = np.nan
        pub_est_ellipse.publish(msg)
        pub_est_error_ellipse.publish(msg)

    ############
    # Method 2 #
    ############
    if arrow_available:
        # method_of_choice = "arrow"
        center_px, radius_length_px, angle_rad = center_px_from_arrow, radius_length_px_from_arrow, angle_from_arrow
        est_arrow_x, est_arrow_y, est_arrow_z = calculate_position(center_px, radius_length_px)
        est_arrow_angle = np.degrees(angle_rad)

        msg.linear.x = est_arrow_x
        msg.linear.y = est_arrow_y
        msg.linear.z = est_arrow_z
        msg.angular.z = est_arrow_angle
        pub_est_arrow.publish(msg)

        try:
            msg.linear.x = est_arrow_x - current_ground_truth[0]
            msg.linear.y = est_arrow_y - current_ground_truth[1]
            msg.linear.z = est_arrow_z - current_ground_truth[2]
            msg.angular.z = est_arrow_angle - current_ground_truth[5]
            pub_est_error_arrow.publish(msg)
        except TypeError as e:
            pass
        draw_dot(global_hsv_canvas_all, center_px, HSV_RED_COLOR, size=7)
    else:
        msg.linear.x = np.nan
        msg.linear.y = np.nan
        msg.linear.z = np.nan
        msg.angular.z = np.nan
        pub_est_arrow.publish(msg)
        pub_est_error_arrow.publish(msg)

    ############
    # Method 3 #
    ############
    if corners_available:
        # method_of_choice = "corners"
        center_px, radius_length_px, angle_rad = center_px_from_inner_corners, radius_px_length_from_inner_corners, angle_from_inner_corners
        est_corners_x, est_corners_y, est_corners_z = calculate_position(center_px, radius_length_px)
        est_corners_angle = np.degrees(angle_rad)
        est_corners_angle = 0.0
        if est_corners_angle < -90:
            est_corners_angle += 180
        elif est_corners_angle > 90:
            est_corners_angle -= 180

        msg.linear.x = est_corners_x
        msg.linear.y = est_corners_y
        msg.linear.z = est_corners_z
        msg.angular.z = est_corners_angle
        pub_est_corners.publish(msg)

        try:
            msg.linear.x = est_corners_x - current_ground_truth[0]
            msg.linear.y = est_corners_y - current_ground_truth[1]
            msg.linear.z = est_corners_z - current_ground_truth[2]
            msg.angular.z = est_corners_angle - current_ground_truth[5]
            pub_est_error_corners.publish(msg)
        except TypeError as e:
            pass
        draw_dot(global_hsv_canvas_all, center_px, HSV_YELLOW_COLOR, size=4)
    else:
        msg.linear.x = np.nan
        msg.linear.y = np.nan
        msg.linear.z = np.nan
        msg.angular.z = np.nan
        pub_est_corners.publish(msg)
        pub_est_error_corners.publish(msg)

    # Choose method #
    if corners_available and green_toughing_edge:
        method_of_choice = 3
        est_x, est_y, est_z = est_corners_x, est_corners_y, est_corners_z
        if arrow_available:
            est_angle = est_arrow_angle
        else:
            est_angle = 0.0
    elif arrow_available:
        method_of_choice = 2
        est_x, est_y, est_z, est_angle = est_arrow_x, est_arrow_y, est_arrow_z, est_arrow_angle
    elif ellipse_available:
        method_of_choice = 1
        est_x, est_y, est_z, est_angle = est_ellipse_x, est_ellipse_y, est_ellipse_z, est_ellipse_angle
    else:
        # method_of_choice = "none"
        method_of_choice = 0
        est_x, est_y, est_z, est_angle = 0.0, 0.0, 0.0, 0.0

    # hsv_save_image(global_hsv_canvas_all, "5_canvas_all_"+str(count))
    hsv_save_image(global_hsv_canvas_all, "5_canvas_all")

    bgr_canvas_all = cv2.cvtColor(global_hsv_canvas_all, cv2.COLOR_HSV2BGR) # convert to HSV

    if draw_on_images:
        try:
            processed_image = bridge.cv2_to_imgmsg(bgr_canvas_all, "bgr8")
        except CvBridgeError as e:
            rospy.loginfo(e)
    else:
        processed_image = None

    result = np.array([est_x, est_y, est_z, 0, 0, est_angle])

    return result, method_of_choice, processed_image


def corner_test():
    # global_image = cv2.imread("0_hsv.png")
    # bgr_angle_test = cv2.imread("corner_angle_test_numerated.png")

    # image_number = 5
    # take_number = 2
    # folder = "./image_processing/still_photos/take_"+str(take_number)+"/"

    # global_image = cv2.imread(folder+"image_"+str(image_number)+".png")
    bgr_angle_test = global_image
    hsv_angle_test = cv2.cvtColor(bgr_angle_test, cv2.COLOR_BGR2HSV)
    # hsv_save_image(hsv_angle_test, "0_hsv_test_image")

    hsv_inside_green = get_pixels_inside_green(hsv_angle_test)
    hsv_h_area = get_h_area(hsv_angle_test)
    hsv_save_image(hsv_h_area, "hsv_h_area")

    bw_angle_test = get_white_mask(hsv_h_area)
    # bw_angle_test = cv2.bitwise_not(bw_angle_test)
    # bw_angle_test = make_gaussian_blurry(bw_angle_test, 7)

    hsv_save_image(bw_angle_test, "bw_angle_test", is_gray=True)
    corners, _ = find_right_angled_corners(bw_angle_test)

    hsv_angle_test_canvas = hsv_angle_test.copy()
    for corner in corners:
        draw_dot(hsv_angle_test_canvas, corner, HSV_RED_COLOR, size=5)
    hsv_save_image(hsv_angle_test_canvas, "angle_test")

    print "Done testing"


def arrow_test():
    image_number = 11

    take_number = 4
    folder = "./image_processing/still_photos/take_"+str(take_number)+"/"

    global_image = cv2.imread(folder+"image_"+str(image_number)+".png")
    bgr_angle_test = global_image
    hsv_arrow_test = cv2.cvtColor(bgr_angle_test, cv2.COLOR_BGR2HSV)
    # hsv_save_image(hsv_arrow_test, "0_hsv_test_image")

    bw_orange_mask = get_orange_mask(hsv_arrow_test)
    if bw_orange_mask is None:
        return None

    # bw_orange_mask = make_gaussian_blurry(bw_orange_mask, 9)
    bw_orange_mask_inverted = cv2.bitwise_not(bw_orange_mask)

    hsv_save_image(bw_orange_mask_inverted, "3_orange_mask_inverted", is_gray=True)

    orange_corners, intensities = find_right_angled_corners(bw_orange_mask_inverted)
    if orange_corners is None:
        return None

    hsv_arrow_test_canvas = hsv_arrow_test.copy()

    for corner in orange_corners:
        draw_dot(hsv_arrow_test_canvas, corner, HSV_RED_COLOR, size=5)
    hsv_save_image(hsv_arrow_test_canvas, "hsv_arrow_test_canvas")

def publish_ground_truth(current_ground_truth):
    global pub_ground_truth

    ground_truth_msg = Twist()
    ground_truth_msg.linear.x = current_ground_truth[0]
    ground_truth_msg.linear.y = current_ground_truth[1]
    ground_truth_msg.linear.z = current_ground_truth[2]
    ground_truth_msg.angular.z = current_ground_truth[5]

    pub_ground_truth.publish(ground_truth_msg)


def main():
    global pub_ground_truth
    global pub_est_ellipse
    global pub_est_arrow
    global pub_est_corners

    global pub_est_error_ellipse
    global pub_est_error_arrow
    global pub_est_error_corners

    global global_ground_truth
    global global_image

    rospy.init_node('cv_module', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, image_callback)
    # rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)
    rospy.Subscriber('/drone_ground_truth', Twist, gt_callback)

    pub_processed_image = rospy.Publisher('/processed_image', Image, queue_size=10)

    pub_est_ellipse = rospy.Publisher("/estimate/ellipse", Twist, queue_size=10)
    pub_est_arrow = rospy.Publisher("/estimate/arrow", Twist, queue_size=10)
    pub_est_corners = rospy.Publisher("/estimate/corners", Twist, queue_size=10)

    pub_est_error_ellipse = rospy.Publisher("/estimate_error/ellipse", Twist, queue_size=10)
    pub_est_error_arrow = rospy.Publisher("/estimate_error/arrow", Twist, queue_size=10)
    pub_est_error_corners = rospy.Publisher("/estimate_error/corners", Twist, queue_size=10)


    pub_est = rospy.Publisher("/estimate/tcv", Twist, queue_size=10)
    pub_est_method = rospy.Publisher("/estimate_method", Int8, queue_size=10)

    est_msg = Twist()

    rospy.loginfo("Starting CV module")



    if not global_is_simulator:
        global_ground_truth = np.zeros(6)


    if use_test_image:
        test_image_filepath = './image_36.png'
        # test_image_filepath = './0_hsv.png'
        global_image = load_bgr_image(test_image_filepath)

        # corner_test()


    count = 0
    rate = rospy.Rate(10) # Hz
    while not rospy.is_shutdown():

        current_ground_truth = global_ground_truth # Fetch the latest ground truth pose available
        if (global_image is not None):
            # denoised = cv2.fastNlMeansDenoisingColored(global_image,None,10,10,7,21) # denoising
            hsv = cv2.cvtColor(global_image, cv2.COLOR_BGR2HSV) # convert to HSV
            est, method, processed_image = get_estimate(hsv, count, current_ground_truth)
            if processed_image is not None:
                pub_processed_image.publish(processed_image)

            pub_est_method.publish(Int8(method))

            if any(est):
                # Publish the estimate
                est_msg.linear.x = est[0]
                est_msg.linear.y = est[1]
                est_msg.linear.z = est[2]
                est_msg.angular.z = est[5]
                pub_est.publish(est_msg)
            count += 1
            if use_test_image:
                break
        else:
            rospy.loginfo("Waiting for image")


        rate.sleep()


if __name__ == '__main__':
    main()
