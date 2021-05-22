#!/usr/bin/env python


"""
PID controller module for use in quadcopter missions, for calculating control signals which will
control the quadcopter to the desired position and orientation.

Subscribes to
    - /filtered_estimate:  quadcopter pose_estimate
    - /set_point: quadcopter desired pose
    - /pid_on_off: toggle pid control
    - /ardrone/takeoff: for internal calculations of control signals
Publishes to
    - /cmd_vel: quadcopter control signal velocities
"""

import rospy
import numpy as np
import help_functions as hlp
from geometry_msgs.msg import Twist, Pose, Point
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Bool
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

import time
import math

import config as cfg



gt_state = None
est_state = None
prev_time = None
pid_on_off = True
prev_setpoint_yaw = None
wf_setpoint = None
bf_setpoint = None



def estimate_callback(data):
    """ Receives estimated pose from pose_estimate package. Estimated pose is body frame.
            data: Twist - estimated pose in body frame
            est_state: float [6]np.array - last received estimated pose in body frames
            bf_setpoint: float [6]np.array - desired pose in body frame
            wf_setpoint: float [6]np.array - desired pose in world frame
    """
    global est_state
    global wf_setpoint
    global bf_setpoint
    global prev_setpoint_yaw

    est_state = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])

    try: #rotate setpoint to match body frame given new yaw
        bf_setpoint[0:2] = hlp.wf_to_bf(wf_setpoint[0:2], est_state[5])
        prev_setpoint_yaw = data.angular.z
    except TypeError as e: # no prev setpoint yaw
        prev_setpoint_yaw = data.angular.z
        pass




def pid_on_off_callback(data):
    """ Toggles PID control on and off depending on std_msgs/Bool data """
    global pid_on_off
    print("PID: ", data.data)
    pid_on_off = data.data

# Setup for the PID controller
error_prev = np.array([0.0]*6)
error_integral = np.array([0.0]*6)
error_derivative = np.array([0.0]*6)
freeze_integral = np.array([False]*6)

wf_setpoint = cfg.default_setpoint
bf_setpoint = [i for i in wf_setpoint]
bf_setpoint[0:2] = hlp.wf_to_bf(wf_setpoint, 0)

# Kp = np.array([Kp_x] + [Kp_y] + [Kp_position_z] + [0.0]*2 + [-Kp_orientation])
Kp = np.array([cfg.Kp_position_x] + [cfg.Kp_position_y] + [cfg.Kp_position_z] + [0.0]*2 + [cfg.Kp_orientation])
Ki = np.array([cfg.Ki_position_x] + [cfg.Ki_position_y] + [cfg.Ki_position_z] + [0.0]*2 + [cfg.Ki_orientation])
Kd = np.array([cfg.Kd_position_x] + [cfg.Kd_position_y] + [cfg.Kd_position_z] + [0.0]*2 + [cfg.Kd_orientation])

actuation_saturation = cfg.actuation_saturation
error_integral_limit = cfg.error_integral_limit


def set_point_callback(data):
    """ Receives new desired pose for the quadcopter, and calculates desired pose in body frame using current estimated yaw.
            data: Twist - desired pose in world frame
            bf_setpoint: float [6]np.array - desired pose in body frame
            wf_setpoint: float [6]np.array - desired pose in world frame
    """
    global wf_setpoint
    global bf_setpoint
    wf_setpoint[0] = data.linear.x # + cfg.offset_setpoint_x
    wf_setpoint[1] = data.linear.y
    wf_setpoint[2] = data.linear.z
    wf_setpoint[5] = data.angular.z
    try:
        bf_setpoint[0:2] = hlp.wf_to_bf(wf_setpoint[0:2],est_state[5])
        bf_setpoint[2] = data.linear.z
        bf_setpoint[5] = data.angular.z
    except TypeError as e:
        bf_setpoint = [i for i in wf_setpoint]

def take_off_callback(data):
    """ Resets error integral on takeoff as error might have accumulated during stand-still.
        data: std_msgs/Empty
    """
    global error_integral
    error_integral = np.array([0.0]*6)

use_gt = False
def toggle_gt_feedback(data):
    global use_gt
    use_gt = not use_gt
    print('Use gt: ', use_gt)

def controller(state):
    """ Calculates the desired control signals (actuation) for the quadcopter in order to reach the desired pose.
    input:
        state: float [6]np.array - the current state of the quadcopter
        bf_setpoint: float [6]np.array - the desired state of the quadcopter
    output:
        actuation_clipped: float [6]np.array - command velocity control signals for quadcopter"""
    global error_prev
    global error_integral
    global error_derivative
    global freeze_integral
    global prev_time

    curr_time = rospy.get_time()
    time_interval = curr_time - prev_time
    prev_time = curr_time
    error = bf_setpoint - state
    error[5] = hlp.angleFromTo(error[5], -180, 180)
    error_integral += (time_interval * (error_prev + error)/2.0)*np.invert(freeze_integral)
    error_derivative = error - error_prev
    error_prev = error

    z_reference = bf_setpoint[2]
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
    global bf_setpoint
    rospy.init_node('pid_controller', anonymous=True)

    if not use_gt:
        rospy.Subscriber('/filtered_estimate', Twist, estimate_callback)
    else:
        rospy.Subscriber('/drone_ground_truth', Twist, estimate_callback)

    rospy.Subscriber('/set_point', Twist, set_point_callback)
    rospy.Subscriber('/ardrone/takeoff', Empty, take_off_callback)
    rospy.Subscriber('/pid_on_off', Bool, pid_on_off_callback)
    rospy.Subscriber('/toggle_gt_feedback', Empty, toggle_gt_feedback)

    control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    time.sleep(1)
    rospy.loginfo("Starting doing PID control with ar_pose as feedback")

    msg = Twist()
    out_of_bounds_error = False
    prev_time = rospy.get_time()
    rate = rospy.Rate(100) # Hz
    while not rospy.is_shutdown():
        state = est_state

        if state is not None and not out_of_bounds_error:
            if abs(state[0]) > cfg.x_upper_limit or abs(state[1]) > cfg.y_upper_limit or abs(state[2]) > cfg.z_upper_limit:
                print('OUT OF BOUNDS AT STATE: ', state)
                out_of_bounds_error = True

        if out_of_bounds_error:
            msg.linear.x = 0
            msg.linear.y = 0
            msg.linear.z = cfg.error_descent_vel
            msg.angular.z = 0
            control_pub.publish(msg)

        elif (state is not None) and pid_on_off:
            actuation = controller(state)
            msg.linear.x = actuation[0]
            msg.linear.y = actuation[1]
            msg.linear.z = actuation[2]
            msg.angular.z = actuation[5]
            control_pub.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    main()
