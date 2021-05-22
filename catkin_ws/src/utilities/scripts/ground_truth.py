#!/usr/bin/env python

"""
This module transforms the ground truth
from world coordinates into body coordinates of the quadcopter

Subscribes to:
    /ground_truth/state: Odometry - world frame ground truth quadcopter Odometry

Publishes to:
    /drone_ground_truth: Twist - body frame ground truth quadcopter pose
"""

import rospy
from geometry_msgs.msg import Twist

import numpy as np

# For ground truth callback:
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R


#################
# ROS functions #
#################
def gt_callback(data):
    global_ground_truth = data.pose.pose

    gt_pose = global_ground_truth
    # Transform ground truth in body frame wrt. world frame to body frame wrt. landing platform

    ##########
    # 0 -> 2 #
    ##########

    # Position
    p_x = gt_pose.position.x
    p_y = gt_pose.position.y
    p_z = gt_pose.position.z

    # Translation of the world frame to body frame wrt. the world frame
    d_0_2 = np.array([p_x, p_y, p_z])

    # Orientation
    q_x = gt_pose.orientation.x
    q_y = gt_pose.orientation.y
    q_z = gt_pose.orientation.z
    q_w = gt_pose.orientation.w

    # Rotation of the body frame wrt. the world frame
    r_0_2 = R.from_quat([q_x, q_y, q_z, q_w])
    r_2_0 = r_0_2.inv()


    ##########
    # 0 -> 1 #
    ##########

    # Translation of the world frame to landing frame wrt. the world frame
    offset_x = 1.0
    offset_y = 0.0
    offset_z = 0.495
    d_0_1 = np.array([offset_x, offset_y, offset_z])

    # Rotation of the world frame to landing frame wrt. the world frame
    # r_0_1 = np.identity(3) # No rotation, only translation
    r_0_1 = np.identity(3) # np.linalg.inv(r_0_1)


    ##########
    # 2 -> 1 #
    ##########
    # Transformation of the body frame to landing frame wrt. the body frame

    # Translation of the landing frame to bdy frame wrt. the landing frame
    d_1_2 = d_0_2 - d_0_1

    # Rotation of the body frame to landing frame wrt. the body frame
    r_2_1 = r_2_0

    yaw = r_2_1.as_euler('xyz')[2]

    r_2_1_yaw = R.from_euler('z', yaw)

    # Translation of the body frame to landing frame wrt. the body frame
    d_2_1 = -r_2_1_yaw.apply(d_1_2)


    # Translation of the landing frame to body frame wrt. the body frame
    # This is more intuitive for the controller
    d_2_1_inv = -d_2_1

    local_ground_truth = np.concatenate((d_2_1_inv, r_2_1.as_euler('xyz')))

    # Transform to get the correct yaw
    yaw = -np.degrees(local_ground_truth[5]) - 90
    if yaw < -180:
        gt_yaw = 360 + yaw
    else:
        gt_yaw = yaw
    local_ground_truth[5] = gt_yaw

    # local_ground_truth


    ground_truth_msg = Twist()
    ground_truth_msg.linear.x = local_ground_truth[0]
    ground_truth_msg.linear.y = local_ground_truth[1]
    ground_truth_msg.linear.z = local_ground_truth[2]
    ground_truth_msg.angular.z = local_ground_truth[5]

    pub_ground_truth.publish(ground_truth_msg)


def main():
    global pub_ground_truth

    rospy.init_node('ground_truth_module', anonymous=True)

    rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)
    pub_ground_truth = rospy.Publisher('/drone_ground_truth', Twist, queue_size=10)

    rospy.loginfo("Starting ground truth module")

    rospy.spin()

if __name__ == '__main__':
    main()
