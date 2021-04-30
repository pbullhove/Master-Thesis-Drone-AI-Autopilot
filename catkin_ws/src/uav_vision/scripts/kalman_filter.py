#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from datetime import datetime
from scipy.spatial.transform import Rotation as R

import time

import config as cfg


freq_yolo = 30
freq_tcv = 10
freq_gps = 1
freq_imu = 100

K_yolo = 0.1*np.eye(6)/freq_yolo
K_tcv = 10*np.eye(6)/freq_tcv
K_gps = 0.5*np.eye(3)/freq_gps
K_imu = 100*np.eye(6)/freq_imu

C_yolo = np.eye(6)
C_tcv = np.eye(6)
C_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
C_yaw = np.array([0,0,0,0,0,1])

def yolo_estimate_callback(data):
    global global_estimate
    yolo_estimate = to_array(data)
    global_estimate = global_estimate + np.dot(K_yolo,(yolo_estimate - np.dot(C_yolo,global_estimate)))

def tcv_estimate_callback(data):
    global global_estimate
    tcv_estimate = to_array(data)
    global_estimate = global_estimate + np.dot(K_tcv,(tcv_estimate - np.dot(C_tcv,global_estimate)))

def gps_callback(data):
    global global_estimate
    gps_estimate = to_array(data)
    gps_estimate = gps_estimate[0:3]
    global_estimate[0:3] = global_estimate[0:3] + np.dot(K_gps,(gps_estimate - np.dot(C_gps,global_estimate)))


calibration_vel = np.array([0.0, 0.0, 0.0])
calibration_acc = np.array([0.0, 0.0, 9.81]) 

vel_min = -0.003
vel_max = 0.003

acc_min = -0.5
acc_max = 0.5

ONE_g = 9.8067

prev_imu_yaw = None
prev_navdata_timestamp = None
def navdata_callback(data):
    global global_estimate
    global prev_imu_yaw
    global prev_navdata_timestamp
    try:
        delta_yaw = data.rotZ - prev_imu_yaw
        now = datetime.now()
        delta_t = (now - prev_navdata_timestamp).total_seconds()
        prev_navdata_timestamp = now
    except TypeError as e: #first iteration
        delta_yaw = 0
        delta_t = 0
        prev_navdata_timestamp = datetime.now()
    finally:
        prev_imu_yaw = data.rotZ


    vel = np.array([data.vx, data.vy, data.vz])/1000 - calibration_vel
    acc = np.array([data.ax*ONE_g, data.ay*ONE_g, data.az*ONE_g]) - calibration_acc
    

    small_values_filter_val = np.logical_and(np.less(vel, vel_max), np.greater(vel, vel_min))
    small_values_filter_acc = np.logical_and(np.less(acc, acc_max), np.greater(acc, acc_min))
    vel[small_values_filter_val] = 0.0
    acc[small_values_filter_acc] = 0.0

    rotation = R.from_euler('z', -np.radians(delta_yaw))

    delta_pos = delta_t*vel + 0.5*acc*delta_t**2
    delta_pos = rotation.apply(delta_pos)

    delta_x = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, delta_yaw])

    global_estimate = global_estimate + np.dot(K_imu,delta_x)

    if global_estimate[5] < -180:
        global_estimate[5] += 360
    elif global_estimate[5] > 180:
        global_estimate[5] -= 360





def to_Twist(array):
    tw = Twist()
    tw.linear.x = array[0]
    tw.linear.y = array[1]
    tw.linear.z = array[2]
    tw.angular.x = array[3]
    tw.angular.y = array[4]
    tw.angular.z = array[5]
    return tw

def to_array(twist):
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr


global_estimate = np.zeros(6)
def main():
    global global_estimate
    rospy.init_node('combined_filter', anonymous=True)

    rospy.Subscriber('/estimate/yolo_estimate', Twist, yolo_estimate_callback)
    rospy.Subscriber('/estimate/tcv_estimate', Twist, tcv_estimate_callback)
    rospy.Subscriber('/mock_gps', Twist, gps_callback)
    rospy.Subscriber('/ardrone/navdata', Navdata, navdata_callback)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)

    rospy.loginfo("Starting combined filter for estimate")


    rate = rospy.Rate(30) # Hz
    while not rospy.is_shutdown():
        global_estimate = [round(i,5) for i in global_estimate]
        msg = to_Twist(global_estimate)
        filtered_estimate_pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    main()
