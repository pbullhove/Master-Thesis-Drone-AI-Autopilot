#!/usr/bin/env python

import rospy
from std_msgs.msg import Empty
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()

global_image_save_path = '/home/peter/Documents/project_thesis/catkin_ws/images/still_photos/'
global_image = None
counter = 0

def image_callback(data):
    global global_image
    #rospy.loginfo('Image callback')
    try:
        #rospy.loginfo('Updated global image')
        global_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
    except CvBridgeError as e:
        rospy.loginfo(e)


def hsv_save_image(image, label='image'):
    global global_image_save_path
    label = label + '.png'
    if not cv2.imwrite(global_image_save_path+label, image):
        raise Exception('Could not save image at location: ', global_image_save_path+label, 'image is:', global_image)


def take_still_photo_callback(data):
    global counter
    if (global_image is not None):
        hsv_save_image(global_image, "image_"+str(counter))
        rospy.loginfo("Still photo taken")
        counter += 1


def main():
    rospy.init_node('still_photos_on_request', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, image_callback)
    rospy.Subscriber('/take_still_photo', Empty, take_still_photo_callback)
    rospy.loginfo("Starting still_photos_on_request module")

    rospy.spin()


def continous_images():
    rospy.init_node('still_photos_continous', anonymous=True)

    rospy.Subscriber('/ardrone/bottom/image_raw', Image, image_callback)
    rospy.Subscriber('/processed_image', Image, image_callback)

    rospy.loginfo("Starting still_photos_continous module")


    rate = rospy.Rate(1) # Hz
    while not rospy.is_shutdown():
        take_still_photo_callback(None)
        rate.sleep()


if __name__ == '__main__':
    main()
    continous_images()
