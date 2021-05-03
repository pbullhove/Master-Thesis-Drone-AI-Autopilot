#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist


def filtered_estimate_callback(data):
    pass

def ground_truth_callback(data):
    pass

def main():
    rospy.init_node("error_calculations", anonymous=True)

    rospy.Subscriber('/filtered_estimate', Twist, filtered_estimate_callback)
    rospy.Subscriber('/drone_ground_truth', Twist, ground_truth_callback)
    filtered_error_publisher = rospy.Publisher('/errors/filtered_estimate_error',Twist, queue_size=10)

    while not rospy.is_shutdown():
        pass



if __name__ == "__main__":
    main()
