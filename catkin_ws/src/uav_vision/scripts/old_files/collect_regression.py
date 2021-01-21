#!/usr/bin/env python
import rospy
import numpy as np
import json

from std_msgs.msg import Empty
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from scipy.spatial.transform import Rotation as R


global_gt_pose = None
global_signal = False


def gt_callback(data):
    global global_gt_pose
    global_gt_pose = data.pose.pose


def signal_callback(data):
    global global_signal
    global_signal = True


def get_relative_position(gt_pose):
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
    offset_z = 0.54
    d_0_1 = np.array([offset_x, offset_y, offset_z])


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
    # Only yaw rotation is considered
    d_2_1 = -r_2_1_yaw.apply(d_1_2)

    # Translation of the landing frame to body frame wrt. the body frame
    # This is more intuitive for the controller
    d_2_1_inv = -d_2_1

    return d_2_1_inv


def main():
    global global_signal

    prev_signal = False
    loaded = True

    rospy.init_node('collect_regression_data', anonymous=True)

    rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)
    rospy.Subscriber('/ardrone/takeoff', Empty, signal_callback)

    rospy.loginfo("Starting collecting regression data")

    filename_dataset = "regression_test.json"
    regression_points = np.empty((0,3))

    rate = rospy.Rate(10) # Hz
    while not rospy.is_shutdown():

        if global_signal and loaded:
            # Do work #
            print("Work")
            if (global_gt_pose is not None):
                position = [get_relative_position(global_gt_pose)]

                regression_points = np.concatenate((regression_points, position))

            dataset = regression_points.tolist()
            
            with open(filename_dataset, 'w') as fp:
                json.dump(dataset, fp)

            ###########

            # Save point
            global_signal = False
            prev_signal = True
            loaded = False
        
        elif global_signal and prev_signal:
            global_signal = False
        
        elif not global_signal and prev_signal:
            print("Reset")
            prev_signal = False
            loaded = True



        

        rate.sleep()
    
    
if __name__ == '__main__':
    main()