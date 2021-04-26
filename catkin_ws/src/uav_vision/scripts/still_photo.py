#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

global_image_save_path = '/home/peter/master_thesis/catkin_ws/images/still_photos/'
global_bottom_image = None
global_front_image = None
front_counter = 0
bottom_counter = 0

def bottom_image_callback(data):
    global global_bottom_image
    #rospy.loginfo('Image callback')
    try:
        #rospy.loginfo('Updated global image')
        global_bottom_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
    except CvBridgeError as e:
        rospy.loginfo(e)

def front_image_callback(data):
    global global_front_image
    #rospy.loginfo('Image callback')
    try:
        #rospy.loginfo('Updated global image')
        global_front_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
    except CvBridgeError as e:
        rospy.loginfo(e)


def hsv_save_image(image, label='image'):
    global global_image_save_path
    label = label + '.png'
    if not cv2.imwrite(global_image_save_path+label, image):
        raise Exception('Could not save image at location: ', global_image_save_path+label, 'image is:', image)


def take_still_photo_front_callback(data):
    global front_counter
    if (global_front_image is not None):
        hsv_save_image(global_front_image, "image_front_"+str(front_counter))
        rospy.loginfo("Still photo taken")
        front_counter += 1

def take_still_photo_bottom_callback(data):
    global bottom_counter
    if (global_bottom_image is not None):
        hsv_save_image(global_bottom_image, "image_bottom_"+str(bottom_counter))
        rospy.loginfo("Still photo taken")
        bottom_counter += 1



def main():
    rospy.init_node('still_photos_on_request', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, bottom_image_callback)
    rospy.Subscriber('/ardrone/front/image_raw', Image, front_image_callback)
    rospy.Subscriber('/take_still_photo_front', Empty, take_still_photo_front_callback)
    rospy.Subscriber('/take_still_photo_bottom', Empty, take_still_photo_bottom_callback)
    rospy.loginfo("Starting still_photos_on_request module. Press 1 and 2 in key teleop for still photos. ")

    rospy.spin()


# def continous_images():
#     rospy.init_node('still_photos_continous', anonymous=True)
#
#     rospy.Subscriber('/ardrone/bottom/image_raw', Image, image_callback)
#
#     rospy.Subscriber('/processed_image', Image, image_callback)
#
#     rospy.loginfo("Starting still_photos_continous module")
#
#
#     rate = rospy.Rate(1) # Hz
#     while not rospy.is_shutdown():
#         take_still_photo_callback(None)
#         rate.sleep()
#

if __name__ == '__main__':
    main()
