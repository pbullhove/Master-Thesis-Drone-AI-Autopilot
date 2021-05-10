#!/usr/bin/env python
"""



"""
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import time
import config as cfg
import sys
sys.path.append('../utilities')
import help_functions as hlp

C_yolo = np.eye(6)
C_tcv = np.eye(6)
C_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
C_sonar = np.array([0,0,1,0,0,0]).reshape((1,6))

R_yolo = 10*np.eye(6)
R_tcv = 0.1*np.eye(6)
R_gps = 1*np.eye(3)
R_sonar = 0.1*np.eye(1)

Q_imu = 0.001*np.diag([1, 1, 1, 0, 0, 1])

P = np.zeros((6,6))

def kalman_gain(P_k,C,R):
    PCT = np.dot(P_k, C.T)
    IS = R + np.dot(C, PCT)
    IS_inv = np.linalg.inv(IS)
    K = np.dot(PCT,IS_inv)
    return K

def P_post(P, C, K):
    P = np.dot((np.eye(6) - np.dot(K,C)), P)
    return P

def P_apri(P, Q):
    P = P + Q
    return P

def KF_update(R,C,y):
    global P
    global x_est
    K = kalman_gain(P, C, R)
    innov = y-np.dot(C,x_est)
    try: # for jumping between yaw = 179, -179
        innov[5] = hlp.angleFromTo(innov[5], -180, 180)
    except IndexError as e:
        pass
    update = np.dot(K, innov)
    x_est = x_est + update
    x_est[5] = hlp.angleFromTo(x_est[5], -180, 180)
    P = P_post(P,C,K)

def yolo_estimate_callback(data):
    """ Filters pose estimates from yolo cv algorithm. Estimates pos in xyz and yaw. Only use this if mmore than 0.7m above platform, as camera view too close for correct estimates. """
    global x_est
    global P
    yolo_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.7:
        if yolo_estimate[5] == 0.0: #if no estimate for yaw
            C = C_gps
            y = yolo_estimate[0:3]
            R = R_yolo[0:3,0:3]
        else:
            C = C_yolo
            R = R_yolo
            y = yolo_estimate
        KF_update(R,C,y)


def tcv_estimate_callback(data):
    """ Filters pose estimates from tcv cv algorithm. Estimates pos in xyz and yaw. Only use this if mmore than 0.4m above platform, as camera view too close for correct estimates. """
    tcv_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.4:
        if tcv_estimate[5] == 0.0 or tcv_estimate[5] == -0.0: #if no estimate for yaw
            C = C_gps
            y = tcv_estimate[0:3]
            R = R_tcv[0:3,0:3]
        else:
            C = C_tcv
            R = R_tcv
            y = tcv_estimate
        KF_update(R,C,y)


def gps_callback(data):
    """ Filters gps data which is measurement of position in xyz. """
    gps_measurement = hlp.twist_to_array(data)
    y = gps_measurement[0:3]
    KF_update(R_gps, C_gps, y)

def sonar_callback(data):
    """ Filters sonar data which is measurement of height. Maximum sonar height is 3m. """
    y = data.range
    if y < 2:
        KF_update(R_sonar, C_sonar, y)

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
    """ Filters estimates from the IMU data and predicts quadcopter pose. """
    global x_est
    global P
    global Q_imu
    global prev_imu_yaw
    global prev_navdata_timestamp
    try:
        delta_yaw = data.rotZ - prev_imu_yaw
        delta_yaw = hlp.angleFromTo(delta_yaw, -180,180)
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


    extreme_values_filter_val = np.logical_and(np.less(vel, vel_max), np.greater(vel, vel_min))
    extreme_values_filter_acc = np.logical_and(np.less(acc, acc_max), np.greater(acc, acc_min))
    vel[extreme_values_filter_val] = 0.0
    acc[extreme_values_filter_acc] = 0.0

    rotation = R.from_euler('z', -np.radians(delta_yaw))

    delta_pos = delta_t*vel + 0.5*acc*delta_t**2

    delta_x = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, delta_yaw])
    x_est = x_est + delta_x

    x_est[0:3] = rotation.apply(x_est[0:3])
    P = P_apri(P, Q_imu)
    x_est[5] = hlp.angleFromTo(x_est[5],-180,180)


x_est = np.zeros(6)
P = np.ones((6,6))
def main():
    global x_est
    rospy.init_node('combined_filter', anonymous=True)

    rospy.Subscriber('/estimate/dnnCV', Twist, yolo_estimate_callback)
    rospy.Subscriber('/estimate/tcv', Twist, tcv_estimate_callback)
    rospy.Subscriber('/mock_gps', Twist, gps_callback)
    rospy.Subscriber('/sonar_height', Range, sonar_callback)
    rospy.Subscriber('/ardrone/navdata', Navdata, navdata_callback)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)

    rospy.loginfo("Starting combined filter for estimate")


    rate = rospy.Rate(30) # Hz
    while not rospy.is_shutdown():
        x_est = [round(i,5) for i in x_est]
        msg = hlp.to_Twist(x_est)
        filtered_estimate_pub.publish(msg)
        rate.sleep()


if __name__ == '__main__':
    main()
