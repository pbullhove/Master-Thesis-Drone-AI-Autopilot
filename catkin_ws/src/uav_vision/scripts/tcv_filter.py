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


est_relative_position = None


def estimate_callback(data):
    global est_relative_position
    est_relative_position = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


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
    global est_relative_position
    rospy.init_node('filter', anonymous=True)

    rospy.Subscriber('/estimate/tcv_estimate', Twist, estimate_callback)
    filtered_estimate_pub = rospy.Publisher('/estimate', Twist, queue_size=10)

    rospy.loginfo("Starting filter for estimate")

    filtered_estimate_msg = Twist()

    # Set up filter
    # median_filter_size = 5
    # average_filter_size = 20
    median_filter_size = 3
    average_filter_size = 3

    estimate_history_size = median_filter_size + average_filter_size - 1
    estimate_history = np.zeros((estimate_history_size,6))

    rate = rospy.Rate(50) # Hz
    while not rospy.is_shutdown():

        if est_relative_position is not None:

            if np.array_equal(est_relative_position, np.zeros(6)):
                est_filtered = np.zeros(6)
            else:
                est_filtered, estimate_history = filter_estimate(est_relative_position, estimate_history, median_filter_size, average_filter_size)

            filtered_estimate_msg.linear.x = est_filtered[0]
            filtered_estimate_msg.linear.y = est_filtered[1]
            filtered_estimate_msg.linear.z = est_filtered[2]
            filtered_estimate_msg.angular.z = est_filtered[5]
            filtered_estimate_pub.publish(filtered_estimate_msg)

            # Mark the estimate as used to avoid filtering the same estimate again
            est_relative_position = None

        rate.sleep()


if __name__ == '__main__':
    main()
