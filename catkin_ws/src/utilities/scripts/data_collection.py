#!/usr/bin/env python

"""
When the 'Triangle'-button on controller is pressed:
The missions start, so collect data

When the 'Square'-button on the controller is pressed:
Stop collecting data and save plot
"""

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool

import time
import matplotlib.pyplot as plt


# Global variables
global_collect_data = False
received_signal = False
start_time = None



def to_array(twist):
    return np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
#############
# Callbacks #
#############

def toggle_data_collection(data):
    global global_collect_data
    global received_signal
    global start_time
    received_signal = True if global_collect_data else False
    global_collect_data = True if not global_collect_data else False
    start_time = rospy.get_time() if start_time == None else start_time


ground_truth = np.zeros(6)
def ground_truth_callback(data):
    global global_est_ground_truth
    ground_truth = to_array(data)

filtered_estimate = np.zeros(6)
filtered_error = np.zeros(6)
def filtered_estimate_callback(data):
    global filtered_estimate
    filtered_estimate = to_array(data)

tcv = np.zeros(6)
tcv_error = np.zeros(6)
def est_tcv_callback(data):
    global tcv
    tcv = to_array(data)

dnnCV = np.zeros(6)
dnnCV_error = np.zeros(6)
def estimate_dnncv_callback(data):
    global dnnCV
    dnnCV = to_array(data)

barometer = np.zeros(6)
barometer_error = np.zeros(6)
def estimate_barometer_callback(data):
    global barometer
    barometer = np.array(data)

gps = np.zeros(6)
gps_error = np.zeros(6)
def estimate_gps_callback(data):
    global gps
    gps = to_array(data)

imu = np.zeros(6)
imu_error = np.zeros(6)
def estimate_imu_callback(data):
    global imu
    imu = to_array(data)

set_point = np.zeros(6)
set_point_error = np.zeros(6)
def set_point_callback(data):
    global set_point
    set_point = to_array(data)




def main():
    global global_state
    global start_time
    rospy.init_node('planner', anonymous=True)

    rospy.Subscriber('/start_data_collection', Empty, toggle_data_collection)
    rospy.Subscriber('/set_point', Twist, set_point_callback)
    rospy.Subscriber('/drone_ground_truth', Twist, ground_truth_callback)
    rospy.Subscriber('/filtered_estimate', Twist, filtered_estimate_callback)

    rospy.Subscriber('/estimate/tcv_estimate', Twist, est_tcv_callback)
    rospy.Subscriber('/estimate/dnnCV', Twist, estimate_dnncv_callback)
    rospy.Subscriber('/estimate/barometer', Twist, estimate_barometer_callback)
    rospy.Subscriber('/estimate/gps', Twist, estimate_gps_callback)
    rospy.Subscriber('/estimate/imu', Twist, estimate_imu_callback)

    pub_heartbeat = rospy.Publisher("/heartbeat_data_collection", Empty, queue_size=10)
    heartbeat_msg = Empty()

    rospy.loginfo("Starting data collection.. Make sure this terminal window is in /master_thesis/.")
    time.sleep(1)
    rospy.loginfo("... Ready! Press d in manual control to start.")

    data_array = []

    duration = 40 # seconds

    rate = rospy.Rate(20) # Hz
    while not rospy.is_shutdown():
        curr_time = rospy.get_time()

        if global_collect_data:
            curr_time = rospy.get_time() - start_time

            set_point_error = set_point - ground_truth
            filtered_error = ground_truth - filtered_estimate
            tcv_error = ground_truth - tcv
            dnnCV_error = ground_truth - dnnCV
            gps_error = ground_truth - gps
            barometer_error = ground_truth - barometer
            imu_error = ground_truth - imu


            data_point = np.concatenate((
                np.array([curr_time]),
                ground_truth,
                set_point,
                filtered_estimate,
                tcv,
                dnnCV,
                gps,
                barometer,
                imu,
                set_point_error,
                filtered_error,
                tcv_error,
                dnnCV_error,
                gps_error,
                barometer_error,
                imu_error
                )
            )
            # print len(data_array)
            print "Time: " + str(curr_time)

            data_array.append(data_point)

            if curr_time > duration:
                break

        if not global_collect_data and received_signal:
            break

        pub_heartbeat.publish(heartbeat_msg)

        rate.sleep()
    folder = './catkin_ws/src/utilities/data_storage/'


    filename = 'unnamed.npy'
    path = folder+filename
    np.save(path, np.array(data_array))


if __name__ == '__main__':
    main()
