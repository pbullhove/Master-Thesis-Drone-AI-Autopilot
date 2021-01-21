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


def detect_moment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_save_image(hsv, '1_hsv')

    img_centroid = hsv.copy()

    white = hsv_keep_white_only(hsv)
    hsv_save_image(white, "2_white", is_gray=True)

    # Threshold image
    _ , img_binary = cv2.threshold(white, 128, 255, cv2.THRESH_BINARY)
    hsv_save_image(img_binary, "3_binary", is_gray=True)

	# calculate moments of binary image
    M = cv2.moments(img_binary)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    cv2.circle(img_centroid, (IMG_WIDTH/2, IMG_HEIGHT/2),
        5, (128, 128, 128), -1)

    cv2.circle(img_centroid, (cX, cY), 5, (255, 255, 255), -1)
    cv2.putText(img_centroid, "centroid", (cX - 25, cY - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    hsv_save_image(img_centroid, "4_centroid")

def run():
    filepath = 'dataset/image_lines_1.jpg'
    # filepath = 'dataset/real_image.png'

    filepaths = [
        'dataset/image_0_corners.jpg',           # 0 corners
        'dataset/image_1_corner.jpg',            # 1 corner
        'dataset/image_lines_1.jpg',             # 2 corners short side
        'dataset/image_lines_3.jpg',             # 3 corners
        'dataset/image_h.jpg',                   # 4 corners
        'dataset/image_2_corners_long_side.jpg', # 2 corners long side
    ]

    img = load_image(filepath)
    detect_moment(img)



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