#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range, ChannelFloat32

def to_array(twist):
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr



filtered_estimate = None
def filtered_estimate_callback(data):
    global filtered_estimate
    filtered_estimate = to_array(data)

ground_truth = None
def ground_truth_callback(data):
    ground_truth = to_array(data)

y_sonar = None
def sonar_measurement_callback(data):
    global y_sonar
    y_sonar = data.range

def main():
    rospy.init_node("error_calculations", anonymous=True)

    rospy.Subscriber('/filtered_estimate', Twist, filtered_estimate_callback)
    rospy.Subscriber('/sonar_height', Range, sonar_measurement_callback)
    rospy.Subscriber('/drone_ground_truth', Twist, ground_truth_callback)
    filtered_error_publisher = rospy.Publisher('/errors/filtered_estimate_error',Twist, queue_size=10)
    sonar_error_publisher = rospy.Publisher('/errors/sonar_error', ChannelFloat32, queue_size=10)

    while not rospy.is_shutdown():
        if ground_truth is not None:
            sonar_error = y_sonar - ground_truth[5]
            print(sonar_error)



if __name__ == "__main__":
    main()
