#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Bool
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

import time
import math

import config as cfg



gt_relative_position = None
est_relative_position = None
prev_time = None
pid_on_off = True

def estimate_callback(data):
    global est_relative_position
    est_relative_position = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


def pid_on_off_callback(data):
    global pid_on_off
    pid_on_off = data.data

# Setup for the PID controller
error_prev = np.array([0.0]*6)
error_integral = np.array([0.0]*6)
error_derivative = np.array([0.0]*6)
freeze_integral = np.array([False]*6)

desired_pose = cfg.controller_desired_pose

# Kp = np.array([Kp_x] + [Kp_y] + [Kp_position_z] + [0.0]*2 + [-Kp_orientation])
Kp = np.array([cfg.Kp_position_x] + [cfg.Kp_position_y] + [cfg.Kp_position_z] + [0.0]*2 + [cfg.Kp_orientation])
Ki = np.array([cfg.Ki_position_x] + [cfg.Ki_position_y] + [cfg.Ki_position_z] + [0.0]*2 + [cfg.Ki_orientation])
Kd = np.array([cfg.Kd_position_x] + [cfg.Kd_position_y] + [cfg.Kd_position_z] + [0.0]*2 + [cfg.Kd_orientation])

actuation_saturation = cfg.actuation_saturation
error_integral_limit = cfg.error_integral_limit


def set_point_callback(data):
    global desired_pose
    desired_pose[0] = data.linear.x # + cfg.offset_setpoint_x
    desired_pose[1] = data.linear.y
    desired_pose[2] = data.linear.z
    desired_pose[5] = data.angular.z


def take_off_callback(data):
    # Reset the error integral each take off,
    # since error might have accumulated during stand-still
    global error_integral
    error_integral = np.array([0.0]*6)


def controller(state):
    global error_prev
    global error_integral
    global error_derivative
    global freeze_integral
    global prev_time

    curr_time = rospy.get_time()
    time_interval = curr_time - prev_time
    prev_time = curr_time

    error = desired_pose - state
    if error[5] < -180:
        error[5] += 360
    error_integral += (time_interval * (error_prev + error)/2.0)*np.invert(freeze_integral)
    error_derivative = error - error_prev
    error_prev = error
    # error_integral = np.clip(error_integral, -error_integral_limit, error_integral_limit)

    z_reference = desired_pose[2]
    actuation_reduction_array = np.array([1.0]*6)
    if z_reference > 2.5:
        actuation_reduction_x_y = np.maximum(2.5 / z_reference, 0.1)
        actuation_reduction_array[0] = actuation_reduction_x_y
        actuation_reduction_array[1] = actuation_reduction_x_y

    actuation = (Kp*error + Kd*error_derivative + Ki*error_integral)*actuation_reduction_array

    actuation_clipped = np.clip(actuation, -actuation_saturation, actuation_saturation)

    # Stop integration when the controller saturates and the system error and the manipulated variable have the same sign
    saturated = np.not_equal(actuation, actuation_clipped)

    dot = error * actuation
    same_sign = np.greater(dot, np.array([0]*6))

    freeze_integral = np.logical_and(saturated, same_sign)


    return actuation_clipped


def main():
    global prev_time

    rospy.init_node('pid_controller', anonymous=True)

    use_estimate = True

    if use_estimate:
        rospy.Subscriber('/estimate/dead_reckoning', Twist, estimate_callback)
    else:
        rospy.Subscriber('/drone_ground_truth', Twist, estimate_callback)


    # rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)

    rospy.Subscriber('/set_point', Twist, set_point_callback)


    rospy.Subscriber('/ardrone/takeoff', Empty, take_off_callback)

    rospy.Subscriber('/pid_on_off', Bool, pid_on_off_callback)


    reference_pub = rospy.Publisher('/drone_reference', Twist, queue_size=10)
    # pose_pub = rospy.Publisher('/drone_pose', Twist, queue_size=10)
    error_pub = rospy.Publisher('/drone_error', Twist, queue_size=10)
    error_integral_pub = rospy.Publisher('/drone_error_integral', Twist, queue_size=10)

    control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    time.sleep(1)

    rospy.loginfo("Starting doing PID control with ar_pose as feedback")

    reference_msg = Twist()
    reference_msg.linear.x = desired_pose[0]
    reference_msg.linear.y = desired_pose[1]
    reference_msg.linear.z = desired_pose[2]
    reference_msg.angular.x = desired_pose[3]
    reference_msg.angular.y = desired_pose[4]
    reference_msg.angular.z = desired_pose[5]

    pose_msg = Twist()
    error_msg = Twist()
    error_integral_msg = Twist()

    prev_time = rospy.get_time()

    rate = rospy.Rate(100) # Hz
    while not rospy.is_shutdown():

        relative_position = est_relative_position
        # if use_estimate:
        #     relative_position = est_relative_position
        # else:
        #     relative_position = gt_relative_position

        if (relative_position is not None) and pid_on_off:
            actuation = controller(relative_position)
            msg = Twist()
            msg.linear.x = actuation[0]
            msg.linear.y = actuation[1]
            msg.linear.z = actuation[2]
            msg.angular.z = actuation[5]

            control_pub.publish(msg)

            # Publish values for tuning
            reference_msg.linear.x = desired_pose[0]
            reference_msg.linear.y = desired_pose[1]
            reference_msg.linear.z = desired_pose[2]
            reference_msg.angular.x = desired_pose[3]
            reference_msg.angular.y = desired_pose[4]
            reference_msg.angular.z = desired_pose[5]
            reference_pub.publish(reference_msg)

            # pose_msg.linear.x = gt_relative_position[0]
            # pose_msg.linear.y = gt_relative_position[1]
            # pose_msg.linear.z = gt_relative_position[2]
            # pose_msg.angular.x = 0
            # pose_msg.angular.y = 0

            # yaw = -np.degrees(gt_relative_position[5]) - 90
            # if yaw < -180:
            #     gt_yaw = 360 + yaw
            # else:
            #     gt_yaw = yaw

            # pose_msg.angular.z = gt_yaw
            # pose_pub.publish(pose_msg)

            error_msg.linear.x = error_prev[0]
            error_msg.linear.y = error_prev[1]
            error_msg.linear.z = error_prev[2]
            error_msg.angular.x = error_prev[3]
            error_msg.angular.y = error_prev[4]
            error_msg.angular.z = error_prev[5]
            error_pub.publish(error_msg)

            error_integral_msg.linear.x = error_integral[0]
            error_integral_msg.linear.y = error_integral[1]
            error_integral_msg.linear.z = error_integral[2]
            error_integral_msg.angular.x = error_integral[3]
            error_integral_msg.angular.y = error_integral[4]
            error_integral_msg.angular.z = error_integral[5]
            error_integral_pub.publish(error_integral_msg)


        rate.sleep()


if __name__ == '__main__':
    main()
