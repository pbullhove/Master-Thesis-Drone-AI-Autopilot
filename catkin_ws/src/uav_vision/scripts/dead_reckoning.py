#!/usr/bin/env python

import rospy
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int8

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import config as cfg

ONE_G = 9.80665

global_velocities = None
global_acceleration = None
global_yaw = None

global_last_position_estimate = None
global_last_yaw_estimate = None

global_estimate_method = 0

string_methods = [
    "None",
    "Ellipse",
    "Arrow",
    "Corners"
]


global_ground_truth = np.zeros(6)
def gt_callback(data):
    global global_ground_truth
    gt_pose = data.pose.pose

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

    global_ground_truth = np.concatenate((d_2_1_inv, r_2_1.as_euler('xyz')))


def estimate_method_callback(data):
    global global_estimate_method
    global_estimate_method = data.data



def navdata_callback(data):
    global global_velocities
    global global_acceleration
    global global_yaw
    global_velocities = np.array([data.vx, data.vy, data.vz])
    global_acceleration = np.array([data.ax*ONE_G*1000, data.ay*ONE_G*1000, data.az*ONE_G*1000])
    # yaw = data.rotZ-90 # Rotation about the Z axis, measured in degrees

    # if yaw < -180:
    #     global_yaw = 360 + yaw
    # else:
    #     global_yaw = yaw

    global_yaw = data.rotZ




def estimate_callback(data):
    global global_last_position_estimate
    global global_last_yaw_estimate

    position = np.array([data.linear.x, data.linear.y, data.linear.z])*1000
    yaw = data.angular.z

    if not np.array_equal(position, np.zeros(3)):
        # Only save last estimate, if there is an estimate available
        global_last_position_estimate = position

    if yaw != 0:
        global_last_yaw_estimate = yaw



def filter_measurement(measurement, measurement_history, median_filter_size, average_filter_size):
    """
        Filters the measurement with a sliding window median and average filter.
    """

    measurement_history = np.concatenate((measurement_history[1:], [measurement]))

    strides = np.array(
        [measurement_history[i:median_filter_size+i] for i in range(average_filter_size)]
    )

    median_filtered = np.median(strides, axis = 1)
    average_filtered = np.average(median_filtered[-average_filter_size:], axis=0)

    return average_filtered, measurement_history


def main():
    global global_last_position_estimate
    global global_last_yaw_estimate
    global global_velocities
    global global_acceleration
    global global_yaw

    rospy.init_node('dead_reckoning', anonymous=True)

    rospy.Subscriber('/ardrone/navdata', Navdata, navdata_callback)
    rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)

    rospy.Subscriber('/estimate', Twist, estimate_callback)
    rospy.Subscriber('/estimate_method', Int8, estimate_method_callback)


    pub_dead_reckoning = rospy.Publisher("filtered_estimate", Twist, queue_size=10)
    pub_dead_reckoning_error = rospy.Publisher("/estimate_error/dead_reckoning", Twist, queue_size=10)


    rospy.loginfo("Starting Dead Reckoning module")
    # rospy.loginfo("Performing initial calibration...")
    time.sleep(1)

    count = 0
    N_CALIBRATION_STEPS = 1000
    calibration_sum_vel = np.zeros(3)
    calibration_sum_acc = np.zeros(3)
    calibration_vel = np.zeros(3)
    calibration_acc = np.zeros(3)

    # Avoid small fluxtuations by removing small values
    vel_min = -3.0
    vel_max = 3.0

    acc_min = -500
    acc_max = 500

    pos_prev = np.zeros(3)
    vel_prev = np.zeros(3)
    yaw_prev = 0.0
    yaw_curr = 0.0

    imu_yaw_prev = None

    # Set up filter
    median_filter_size = 1
    average_filter_size = 3

    history_size = median_filter_size + average_filter_size - 1
    vel_history = np.zeros((history_size,3))
    acc_history = np.zeros((history_size,3))

    dead_reckoning_msg = Twist()

    do_calibration_before_start = cfg.do_calibration_before_start

    rate = rospy.Rate(100) # Hz
    while not rospy.is_shutdown():
        if (global_velocities is not None) and (global_acceleration is not None) and (global_yaw is not None):
            imu_vel_new = global_velocities
            imu_acc_new = global_acceleration
            imu_yaw_new = global_yaw
            if imu_yaw_prev is None:
                imu_yaw_prev = imu_yaw_new # Set prev value the first time
            global_velocities, global_acceleration, global_yaw = None, None, None # Reset global variables to only use one measurement once
            if count == 0:
                start_time = rospy.get_time()
                prev_time = start_time

                if not do_calibration_before_start:
                    calibration_vel = np.array([0.0, 0.0, 0.0])
                    calibration_acc = np.array([0.0, 0.0, 9.81e+03])

            elif count < N_CALIBRATION_STEPS and do_calibration_before_start:
                calibration_sum_vel += imu_vel_new
                calibration_sum_acc += imu_acc_new
            elif count == N_CALIBRATION_STEPS and do_calibration_before_start:
                calibration_vel = calibration_sum_vel / float(N_CALIBRATION_STEPS)
                calibration_acc = calibration_sum_acc / float(N_CALIBRATION_STEPS)

                end_time = rospy.get_time()
                duration = end_time - start_time
                rospy.loginfo("Calibration ready. Duration: " + str(duration))
                prev_time = end_time
            else: # Perform dead reckoning
                # rospy.loginfo("Method: " + string_methods[global_estimate_method])

                # Calculate delta yaw from IMU,
                # convert large step in oppisite direction
                # to a small step in same direction
                delta_imu_yaw = imu_yaw_new - imu_yaw_prev
                if delta_imu_yaw > 180:
                    delta_imu_yaw -= 360
                elif delta_imu_yaw < -180:
                    delta_imu_yaw += 360
                imu_yaw_prev = imu_yaw_new

                # Position
                if global_last_position_estimate is not None:
                    # Use last estimate if available
                    pos_curr = global_last_position_estimate
                    global_last_position_estimate = None
                else:
                    curr_time = rospy.get_time()
                    duration = curr_time - prev_time
                    prev_time = curr_time

                    vel = imu_vel_new - calibration_vel
                    acc = imu_acc_new - calibration_acc

                    # vel, vel_history = filter_measurement(vel, vel_history, median_filter_size, average_filter_size)
                    # acc, acc_history = filter_measurement(acc, acc_history, median_filter_size, average_filter_size)

                    small_values_filter_val = np.logical_and(np.less(vel, vel_max), np.greater(vel, vel_min))
                    small_values_filter_acc = np.logical_and(np.less(acc, acc_max), np.greater(acc, acc_min))
                    vel[small_values_filter_val] = 0.0
                    acc[small_values_filter_acc] = 0.0

                    rotation = R.from_euler('z', -np.radians(delta_imu_yaw))

                    delta_position = duration*vel + 0.5*acc*duration**2
                    delta_position = rotation.apply(delta_position)

                    pos_curr = pos_prev + delta_position


                # Yaw
                if global_last_yaw_estimate is not None:
                    yaw_curr = global_last_yaw_estimate
                    global_last_yaw_estimate = None
                else:
                    yaw_curr = yaw_prev + delta_imu_yaw

                    if yaw_curr > 180:
                        yaw_curr -= 360
                    elif yaw_curr < -180:
                        yaw_curr += 360


                pos_prev = pos_curr
                yaw_prev = yaw_curr

                dead_reckoning_x = pos_curr[0] / 1000
                dead_reckoning_y = pos_curr[1] / 1000
                dead_reckoning_z = pos_curr[2] / 1000
                dead_reckoning_yaw = yaw_curr

                dead_reckoning_msg.linear.x = dead_reckoning_x
                dead_reckoning_msg.linear.y = dead_reckoning_y
                dead_reckoning_msg.linear.z = dead_reckoning_z
                dead_reckoning_msg.angular.z = dead_reckoning_yaw
                pub_dead_reckoning.publish(dead_reckoning_msg)

                dead_reckoning_msg.linear.x = dead_reckoning_x - global_ground_truth[0]
                dead_reckoning_msg.linear.y = dead_reckoning_y - global_ground_truth[1]
                dead_reckoning_msg.linear.z = dead_reckoning_z - global_ground_truth[2]
                dead_reckoning_msg.angular.z = dead_reckoning_yaw - global_ground_truth[5]
                pub_dead_reckoning_error.publish(dead_reckoning_msg)

            count += 1
        rate.sleep()


if __name__ == '__main__':
    main()
