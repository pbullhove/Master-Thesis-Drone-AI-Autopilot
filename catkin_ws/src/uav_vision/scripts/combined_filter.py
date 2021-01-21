#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

import time
import math

import config as cfg


yolo_estimate = None
def yolo_estimate_callback(data):
    global yolo_estimate
    yolo_estimate = np.array([data.linear.x, data.linear.y, data.linear.z, data.angular.x, data.angular.y, data.angular.z])

tcv_estimate = None
def tcv_estimate_callback(data):
    global tcv_estimate
    tcv_estimate = np.array([data.linear.x, data.linear.y, data.linear.z, data.angular.x, data.angular.y, data.angular.z])


def filter_estimate(estimate, estimate_history, median_filter_size, average_filter_size):
    """
        Filters the estimate with a sliding window median and average filter.
    """

    estimate_history = np.concatenate((estimate_history[1:], [estimate]))

    strides = np.array(
        [estimate_history[i:median_filter_size+i] for i in range(average_filter_size)]
    )

    median_filtered = np.median(strides, axis = 1)
    average_filtered = np.average(median_filtered[-average_filter_size:], axis=0)

    return average_filtered, estimate_history


def main():
    global tcv_estimate
    global yolo_estimate
    rospy.init_node('combined_filter', anonymous=True)

    rospy.Subscriber('/estimate/yolo_estimate', Twist, yolo_estimate_callback)
    rospy.Subscriber('/estimate/tcv_estimate', Twist, tcv_estimate_callback)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)
    filtered_estimate_msg = Twist()

    rospy.loginfo("Starting combined filter for estimate")

    average_filter_size = 3
    median_filter_size = 3

    estimate_history_size = median_filter_size + average_filter_size - 1
    estimate_history = np.zeros((estimate_history_size,6))

    est_filtered = np.zeros(6)
    est_filtered_prev = None
    rate = rospy.Rate(50) # Hz
    while not rospy.is_shutdown():

        if tcv_estimate is not None:
            est_filtered, estimate_history = filter_estimate(tcv_estimate, estimate_history, median_filter_size, average_filter_size)
            tcv_estimate = None

        if yolo_estimate is not None:
            if -0.1 < yolo_estimate[5] < 0.1:
                yolo_estimate[5] = np.average(estimate_history[:,5])
            est_filtered, estimate_history = filter_estimate(yolo_estimate, estimate_history, median_filter_size, average_filter_size)
            yolo_estimate = None

        if not (est_filtered == est_filtered_prev).all():
            filtered_estimate_msg.linear.x = est_filtered[0]
            filtered_estimate_msg.linear.y = est_filtered[1]
            filtered_estimate_msg.linear.z = est_filtered[2]
            filtered_estimate_msg.angular.z = est_filtered[5]
            filtered_estimate_pub.publish(filtered_estimate_msg)
            est_filtered_prev = est_filtered
        rate.sleep()


if __name__ == '__main__':
    main()
