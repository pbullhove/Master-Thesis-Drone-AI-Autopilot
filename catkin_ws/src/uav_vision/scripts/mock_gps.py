#!/usr/bin/env python
import rospy
import time
import numpy as np

from geometry_msgs.msg import Twist

#############
# Callbacks #
#############
ground_truth = None
def gt_callback(data):
    global ground_truth
    ground_truth = data


def generate_random_errors(standard_deviation):
    mu = 0
    xyz = np.random.normal(mu, standard_deviation, 3)
    return xyz

def main():
    rospy.init_node('mock_gps', anonymous=True)


    rospy.Subscriber('/drone_ground_truth', Twist, gt_callback)
    pub_gps = rospy.Publisher('/mock_gps', Twist, queue_size=1)
    pub_msg = Twist()

    sigma = 0.02

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        if ground_truth is not None:
            ex,ey,ez = generate_random_errors(sigma)
            pub_msg.linear.x = round(ground_truth.linear.x + ex, 3)
            pub_msg.linear.y = round(ground_truth.linear.y + ey, 3)
            pub_msg.linear.z = round(ground_truth.linear.z + ez, 3)
            pub_gps.publish(pub_msg)
        else:
            print('No gt data received yet, start ground_truth.py')
        rate.sleep()


if __name__ == "__main__":
    main()
