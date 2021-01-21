#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image


import matplotlib.pyplot as plt
import numpy as np
import math

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()
drone_image_raw = None

IMG_WIDTH = 640
IMG_HEIGHT = 360

# For converting between units
conv_scale_to_bits = np.array([0.5, 2.55, 2.55]) # unit bits with ranges [[0,180], [0,255], [0,255]]


#############
# Callbacks #
#############

def video_callback(data):
    global drone_image_raw
    drone_image_raw = data


######################
# Help functions #
######################
def make_blurry(image, blur):
    return cv2.medianBlur(image, blur)


def make_gaussian_blurry(image, blur):
    return cv2.GaussianBlur(image, (blur, blur), 0)


def hsv_save_image(image, label='image', is_gray=False):
    folder = 'image_processing/detect_h/'
    if is_gray:
        cv2.imwrite(folder+label+".png", image)
    else:
        cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))
    return image


def load_image(filename):
    img = cv2.imread(filename) # import as BGR
    return img


def rgb_color_to_hsv(red, green, blue):
    bgr_color = np.uint8([[[blue,green,red]]])
    hsv_color = cv2.cvtColor(bgr_color,cv2.COLOR_BGR2HSV)
    return hsv_color


def normalize_vector(vector):
    return vector / np.linalg.norm(vector) #, ord=1)


##################
# Main functions #
##################
def hsv_keep_white_only(hsv):
    # In degrees, %, %, ranges [[0,360], [0,100], [0,100]]
    lower_white_all = np.array([  0,   0,  90])*conv_scale_to_bits
    upper_white_all = np.array([360,  20, 100])*conv_scale_to_bits

    # In degrees, %, %, ranges [[0,360], [0,100], [0,100]]
    lower_white_except_orange = np.array([ 70,   0,  85])*conv_scale_to_bits
    upper_white_except_orange = np.array([360,  40, 100])*conv_scale_to_bits

    # In degrees, %, %, ranges [[0,360], [0,100], [0,100]]
    lower_white_almost_green = np.array([ 0,   0,  85])*conv_scale_to_bits
    upper_white_almost_green = np.array([70,  10, 100])*conv_scale_to_bits

    # In degrees, %, %, ranges [[0,360], [0,100], [0,100]]
    lower = np.array([  0,   0,  70])*conv_scale_to_bits
    upper = np.array([ 50,  20, 100])*conv_scale_to_bits

    mask_all = cv2.inRange(hsv, lower_white_all, upper_white_all)
    mask_except_orange = cv2.inRange(hsv, lower_white_except_orange, upper_white_except_orange)
    mask_except_almost_green = cv2.inRange(hsv, lower_white_almost_green, upper_white_almost_green)
    mask = cv2.inRange(hsv, lower, upper) # May be to detect all the white

    combined_mask = mask_all + mask_except_orange + mask_except_almost_green # + mask

    return combined_mask


def est_position(mid_point):
    """ Not finished implementing """
    print "Mid point:", mid_point


def est_yaw(dir_vector):
    """ Not finished implementing """
    print "Direction vector:", dir_vector
    dir_vector_normalized = normalize_vector(dir_vector)
    v_dir = dir_vector_normalized
    print "Direction vector normalized:", dir_vector_normalized

    ref_vector = np.array([-1, 0]) # Vector pointing straight up
    v_ref = ref_vector


    v_dir = normalize_vector(np.array([-3, -2]))



    prod = np.dot(v_dir, v_ref)

    alpha = np.arccos(np.clip(prod, -1.0, 1.0))

    print "Angle:", np.degrees(alpha)

    dot = v_ref[0]*v_ref[1] + v_dir[0]*v_dir[1],
    det = v_ref[0]*v_dir[1] - v_dir[0]*v_ref[1]

    signed_angle = np.arctan2(det, dot)
    print "Signed angle:", np.degrees(signed_angle)

# a = atan2d(x1*y2-y1*x2,x1*x2+y1*y2);


def detect_sub_pixecl_corners(img, img_id = 0):
    # Parameters ###########################
    second_blur_size = 49
    ignore_border_size = 3
    corner_harris_block_size = 4

    # Define valid intensity range for the median of a corner
    min_intensity_average = 170
    max_intensity_average = 240
    ########################################

    hsv_origin = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv_origin
    hsv = hsv_save_image(hsv_origin, '1_hsv')

    hsv_red_color = rgb_color_to_hsv(255, 0, 0)
    
    white = hsv_keep_white_only(hsv)
    hsv_save_image(white, "2_white", is_gray=True)

    blur = make_gaussian_blurry(white, 5) 
    double_blur = make_gaussian_blurry(blur, second_blur_size)
    # blur = hsv_save_image(blur, "3a_blur", is_gray=True)
    # double_blur = hsv_save_image(double_blur, "3b_double_blur", is_gray=True)
    
    # Sub-pixel:
    dst = cv2.cornerHarris(blur,corner_harris_block_size,3,0.04)
    dst = cv2.dilate(dst,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(blur,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    img_all_corners = hsv_origin.copy()
    img_best_corners = hsv_origin.copy()
    
    number_of_corners = len(corners)
    x_min = np.array([ignore_border_size]*number_of_corners)
    x_max = np.array([IMG_HEIGHT - ignore_border_size]*number_of_corners)
    y_min = np.array([ignore_border_size]*number_of_corners)
    y_max = np.array([IMG_WIDTH - ignore_border_size]*number_of_corners)

    # Keep corners within the border limit
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
    img_all_corners[corner_x, corner_y] = hsv_red_color
    
    number_of_corners = len(corner_x)
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
    img_best_corners[corner_x, corner_y] = hsv_red_color

    # Save images with marked corners
    # hsv_save_image(img_all_corners, '4_all_corners')
    # hsv_save_image(img_best_corners, '4_best_corners')


    if np.ndim(corner_x) == 0:
        number_of_corners = 1
        corners = np.array([[corner_x, corner_y]])
    else:
        corners = np.stack((corner_x, corner_y), axis=1)
        number_of_corners = len(corners)

    print "Number of corners:", number_of_corners


    if number_of_corners == 0 or number_of_corners > 4:
        print("Invalid number of corners")
        return None


    ######################
    # Define the corners #

    if number_of_corners == 1:
        corner_0 = corners[0]

    if number_of_corners == 2:
        left_corner_id = np.argmin(corner_y)
        right_corner_id = 1 - left_corner_id

        corner_0 = corners[left_corner_id]
        corner_1 = corners[right_corner_id]


    if number_of_corners == 3:
        distances = np.array([
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[1] - corners[2]),
            np.linalg.norm(corners[2] - corners[0])
        ])

        min_dist_id = np.argmin(distances)

        if min_dist_id == 0:
            relevant_corners = np.stack((corners[0], corners[1]), axis=0)
        elif min_dist_id == 1:
            relevant_corners = np.stack((corners[1], corners[2]), axis=0)
        elif min_dist_id == 2:
            relevant_corners = np.stack((corners[2], corners[0]), axis=0)
        else:
            print("ERROR: In fn. detect_sub_pixecl_corners(); min_dist_id out of bounds")
            return None

        dist_0_1, dist_1_2, dist_2_0 = distances[0], distances[1], distances[2]

        # Pick the corner with the lowest y-coordinate
        left_corner_id = np.argmin(relevant_corners, axis=0)[1]
        right_corner_id = 1 - left_corner_id

        corner_0 = relevant_corners[left_corner_id]
        corner_1 = relevant_corners[right_corner_id]

        img_corners = hsv_origin.copy()
        img_corners[corner_0[0]][corner_0[1]] = hsv_red_color
        img_corners[corner_1[0]][corner_1[1]] = hsv_red_color
        hsv_save_image(img_corners, "5_relevant_corners")


    if number_of_corners == 4:
        # For the first corner, chose the corner closest to the top
        # This will belong to the top cross-bar
        top_corner_id = np.argmin(corner_x)
        top_corner = corners[top_corner_id]
        top_corner_stack = np.array([top_corner]*3)
        rest_corners = np.delete(corners, top_corner_id, 0)
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

        # Draw the corners
        img_corners = hsv_origin.copy()
        img_corners[corner_1[0]][corner_1[1]] = hsv_red_color
        # hsv_save_image(img_corners, "5_relevant_corners")


    ##################################
    # Define mid point and direction #'
    mid_point = None
    dir_vector = None

    # Calculate spatial gradient
    dy, dx	= cv2.spatialGradient(blur) # Invert axis, since OpenCV operates with x=column, y=row
    
    if number_of_corners == 1:
        grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
        grad_0_normalized = normalize_vector(grad_0)

        mid_point_offset = 5 # px-distance
        mid_point = np.int0(corner_0 + grad_0_normalized*mid_point_offset)

        dir_vector = grad_0_normalized

    else:
        grad_0 = np.array([dx[corner_0[0]][corner_0[1]], dy[corner_0[0]][corner_0[1]]])
        grad_1 = np.array([dx[corner_1[0]][corner_1[1]], dy[corner_1[0]][corner_1[1]]])
        grad_sum = grad_0 + grad_1
        grad_sum_normalized = normalize_vector(grad_sum)

        cross_over_vector = corner_1 - corner_0
        mid_point = np.int0(corner_0 + 0.5*(cross_over_vector))
        mid_value = double_blur[mid_point[0]][mid_point[1]]


    if number_of_corners == 2 and mid_value < 200:
        print "The two corners form the longitudinal bar"
        # Choose to focus on the left point

        v_long = corner_0 - corner_1
        v_long_norm = normalize_vector(v_long)

        grad_0_normalized = normalize_vector(grad_0)

        if grad_0_normalized[0] < 0:
            long_bar_norm = np.array([-cross_over_vector[1], cross_over_vector[0]])
        else:
            long_bar_norm = np.array([cross_over_vector[1], -cross_over_vector[0]])
        long_bar_norm_normalized = normalize_vector(long_bar_norm)
        

        dist_to_top_edge = corner_0[0]
        dist_to_bottom_edge = IMG_HEIGHT - corner_0[0]
        dist_to_nearest_edge = min(dist_to_top_edge, dist_to_bottom_edge)
        mid_point_offset = dist_to_nearest_edge / 2 # px-distance
        mid_point = np.int0(corner_0 + long_bar_norm_normalized*mid_point_offset)

        dir_vector = v_long_norm
    
    elif number_of_corners != 1:
        if grad_sum_normalized[0] < 0:
            cross_over_norm = np.array([-cross_over_vector[1], cross_over_vector[0]])
        else:
            cross_over_norm = np.array([cross_over_vector[1], -cross_over_vector[0]])

        dir_vector = normalize_vector(cross_over_norm)


    ###################
    # Draw the result #

    direction_length = 20
    direction_point = np.int0(mid_point + dir_vector*direction_length)

    img_grad = hsv_origin.copy()
    img_grad = cv2.arrowedLine(img_grad, (mid_point[1], mid_point[0]),
        (direction_point[1], direction_point[0]),
        color = (255,0,0), thickness = 2, tipLength = 0.5)

    # hsv_save_image(img_grad, "5_gradient")
    hsv_save_image(img_grad, str(img_id) +"_gradient")

    est_position(mid_point)
    est_yaw(dir_vector)


    return img_grad


def run():
    filepath = 'dataset/image_1_corner.jpg'
    # filepath = 'dataset/real_image.png'

    filepaths = [
        'dataset/image_0_corners.jpg',           # 0 corners
        'dataset/image_1_corner.jpg',            # 1 corner
        'dataset/image_lines_1.jpg',             # 2 corners short side
        'dataset/image_lines_3.jpg',             # 3 corners
        'dataset/image_h.jpg',                   # 4 corners
        'dataset/image_2_corners_long_side.jpg', # 2 corners long side
    ]

    img_id = 4


    for img_id in range(1,6):
        img = load_image(filepaths[img_id])
        detect_sub_pixecl_corners(img, img_id)
        print


""" TODO:
    # Use the found corners to estimate position in x, y and yaw
    x_est = None
    y_est = None
    yaw_est = None
"""

run()


def main():   
    
    rospy.init_node('detect_h', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, video_callback)

    pub_image = rospy.Publisher('/image_with_direction', Image, queue_size=10)

    rospy.loginfo("Starting preprocessing of image")

    rate = rospy.Rate(100) # Hz
    while not rospy.is_shutdown():


        
        if drone_image_raw is not None:
            try:
                cv_image = bridge.imgmsg_to_cv2(drone_image_raw, 'bgr8') # , 'bgr8', 'rgb8
            except CvBridgeError as e:
                rospy.loginfo(e)
        
            cv_img_with_direction = detect_sub_pixecl_corners(cv_image)
            if cv_img_with_direction is not None:
                cv_bgr_img_with_direction = cv2.cvtColor(cv_img_with_direction, cv2.COLOR_HSV2BGR)

                try:
                    msg_img_with_direction = bridge.cv2_to_imgmsg(cv_bgr_img_with_direction, 'bgr8')
                except CvBridgeError as e:
                    rospy.loginfo(e)


                pub_image.publish(msg_img_with_direction)
            
        rate.sleep()


# if __name__ == '__main__':
#     main()