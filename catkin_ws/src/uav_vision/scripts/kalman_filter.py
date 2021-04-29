#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from datetime import datetime
import time

import config as cfg


freq_yolo = 30
freq_tcv = 10
freq_gps = 1
freq_imu = 100

K_yolo = 0.1*np.eye(6)/freq_yolo
K_tcv = 10*np.eye(6)/freq_tcv
K_gps = 0.5*np.eye(3)/freq_gps
K_yaw = 0.1*np.eye(1)/freq_imu

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



prev_imu_yaw = None
prev_navdata_timestamp = None
def navdata_callback(data):
    global global_estimate
    global prev_imu_yaw
    global prev_navdata_timestamp
    try:
        delta_imu_yaw = data.rotZ - prev_imu_yaw
        now = datetime.now()
        delta_t = (now - prev_navdata_timestamp).total_seconds()
        prev_navdata_timestamp = now
    except TypeError as e: #first iteration
        delta_imu_yaw = 0
        delta_t = 0
        prev_navdata_timestamp = datetime.now()
    finally:
        prev_imu_yaw = data.rotZ


    global_estimate[5] = global_estimate[5] + delta_imu_yaw
    if global_estimate[5] < -180:
        global_estimate[5] += 360
    elif global_estimate[5] > 180:
        global_estimate[5] -= 360


def to_Twist(data):
    msg = Twist()
    msg.linear.x = data[0]
    msg.linear.y = data[1]
    msg.linear.z = data[2]
    msg.angular.x = data[3]
    msg.angular.y = data[4]
    msg.angular.z = data[5]
    return msg

def to_array(twist):
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr


def to_body_frame(data)


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
