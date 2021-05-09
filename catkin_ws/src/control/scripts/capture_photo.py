#!/usr/bin/env python

"""
Module for saving still photos from quadcopter missions.
Saves photos in ~/master_thesis/images/mission_photos

Subscribes to:
    /ardrone/front/image_raw: Image - the camera feed from the front camera on the ar.drone
    /ardrone/bottom/image_raw: Image - the camera feed from the bottom camera on the ar.drone
    /take_still_photo_front: Empty - command to save image from front camera
    /take_still_photo_bottom: Empty - command to save image from bottom camera

Publishes to:
    None.
"""


import rospy
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
from datetime import datetime

bridge = CvBridge()
try:
    os.chdir('../../../..')
    os.chdir('images/mission_photos')
    print(os.getcwd())
except Exception as e:
    pass
front_counter = 0
bottom_counter = 0

bottom_capture_request = False
front_capture_request = False


def front_image_callback(data):
    """
    Continuously receives images from front camera.
    If front_capture_request is True:
        Then saves the next image that comes from the camera.
        Saves images in folder ~/master_thesis/images/mission_photos
    front_capture_request is toggled by topic /take_still_photo_front
    """
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
    """
    Continuously receives images from bottom camera.
    If bottom_capture_request is True:
        Then saves the next image that comes from the camera.
        Saves images in folder ~/master_thesis/images/mission_photos
    bottom_capture_request is toggled by topic /take_still_photo_bottom
    """
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
    """ Toggles front_capture_request such that next front image is captured. """
    global front_capture_request
    front_capture_request = True


def take_still_photo_bottom_callback(data):
    """ Toggles bottom_capture_request such that next bottom image is captured. """
    global bottom_capture_request
    bottom_capture_request = True


def hsv_save_image(image, label='image'):
    """
    Saves image to current location: check that by os.getcdw()
    """
    global global_image_save_path
    label = label + '.png'
    path = label
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
