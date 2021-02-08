#!/usr/bin/env python
import rospy
import numpy as np


x = np.matrix('')
P = np.matrix('')

A = np.matrix('')
B = np.matrix('')

C_imu = np.matrix('')
C_yolo = np.matrix('')
C_tcv = np.matrix('')

Q = np.matrix('')

R_imu = np.matrix('')
R_yolo = np.matrix('')
R_tcv = np.matrix('')


def imu_measurement_callback(data):
    global x, P
    y = data
    x, P = kf_update(x, P, y, C_imu, R_imu)

def tcv_measurement_callback(data):
    global x, P
    y = data
    x, P = kf_update(x, P, y, C_tcv, R_tcv)

def yolo_measurement_callback(data):
    global x, P
    y = data
    x, P = kf_update(x, P, y, C_yolo, R_tcv)



def kf_predict(x, P, A, Q, B, u):
    x = np.dot(A,x) + np.dot(B,U)
    P = np.dot(A, np.dot(P, A.T)) + Q
    return (x, P)


def kf_update(x, P, y, C, R):
    IM = np.dot(C, x)
    IS = R + np.dot(C, np.dot(P, C.T))
    K = np.dot(P, np,dot(C.T, np.inv(IS)))
    x = x + np.dot(K, (Y-IM))
    P = P - np.dot(K, np.dot(IS, K.T))
    return (x, P)



if __name__ == "__main__":
    rospy.init_node('kalman_filter', anonymous=True)
    rospy.Subscriber('/measurement/imu', Twist, imu_measurement_callback)
    rospy.Subscriber('/measurement/yolo', Twist, yolo_measurement_callback)
    rospy.Subscriber('/measurement/tcv', Twist, tcv_measurement_callback)

    kf_publisher = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)
    kf_estimate = Twist()

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():

        rate.sleep()
