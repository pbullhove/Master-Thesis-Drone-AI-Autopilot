#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import os


bridge = CvBridge()

global_image_save_path = os.path.join('master_thesis','images','mission_photos')

front_counter = 0
bottom_counter = 0

bottom_capture_request = False
front_capture_request = False


def front_image_callback(data):
    global front_counter
    global front_capture_request
    if front_capture_request:
        try:
            #rospy.loginfo('Updated global image')
            front_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
            if (front_image is not None):
                hsv_save_image(front_image, "image_front_"+str(front_counter))
                rospy.loginfo("Front photo captured")
                front_counter += 1
        except CvBridgeError as e:
            rospy.loginfo(e)
        finally:
            front_capture_request = False


def bottom_image_callback(data):
    global bottom_counter
    global bottom_capture_request
    if bottom_capture_request:
        try:
            #rospy.loginfo('Updated global image')
            bottom_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
            if (bottom_image is not None):
                hsv_save_image(bottom_image, "image_bottom_"+str(bottom_counter))
                rospy.loginfo("Bottom photo captured")
                bottom_counter += 1

        except CvBridgeError as e:
            rospy.loginfo(e)

        finally: 
                bottom_capture_request = False


def take_still_photo_front_callback(data):
    global front_capture_request
    front_capture_request = True


def take_still_photo_bottom_callback(data):
    global bottom_capture_request
    bottom_capture_request = True


def hsv_save_image(image, label='image'):
    global global_image_save_path
    label = label + '.png'
    path = os.path.join(global_image_save_path, label)
    if not cv2.imwrite(path, image):
        raise Exception('Could not save image at location: ', path, 'image is:', image)


def main():
    rospy.init_node('still_photos_on_request', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, bottom_image_callback)
    rospy.Subscriber('/ardrone/front/image_raw', Image, front_image_callback)
    rospy.Subscriber('/take_still_photo_front', Empty, take_still_photo_front_callback)
    rospy.Subscriber('/take_still_photo_bottom', Empty, take_still_photo_bottom_callback)
    rospy.loginfo("Starting still_photos_on_request module. Press 1 and 2 in key teleop for still photos from bottom and front camera. ")

    rospy.spin()

if __name__ == '__main__':
    main()