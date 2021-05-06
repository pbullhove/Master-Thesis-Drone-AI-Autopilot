#!/usr/bin/env python
import math
import rospy
import numpy as np
from geometry_msgs.msg import Twist

def to_Twist(array):
    tw = Twist()
    tw.linear.x = array[0]
    tw.linear.y = array[1]
    tw.linear.z = array[2]
    tw.angular.x = array[3]
    tw.angular.y = array[4]
    tw.angular.z = array[5]
    return tw

def twist_to_array(twist):
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr


def wf_to_bf(wf,yaw):
    yaw *= math.pi/180
    c = math.cos(yaw)
    s = math.sin(yaw)
    r_inv = np.array([[c, s, 0],[-s, c, 0],[0,0,1]])

    wf = np.array([wf[0], wf[1], 1])
    bf = np.dot(r_inv,wf)
    return bf[0:2]


def angleFromTo(ang, min, max):
    if ang < min:
        ang += 360
    if ang > max:
        ang -= 360
    return ang



def bf_to_wf(bf,yaw):
    yaw *= math.pi/180
    c = math.cos(yaw)
    s = math.sin(yaw)
    r = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

    bf = np.array([bf[0], bf[1], 1])
    wf = np.dot(r,bf)
    return bf[0:2]


def twist_bf_to_wf(bf):
    yaw = bf.angular.z
    yaw *= math.pi/180
    c = math.cos(yaw)
    s = math.sin(yaw)
    r = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

    wf = Twist()
    xy = np.array([bf.linear.x, bf.linear.y, 1])
    wf.linear.x, wf.linear.y = np.dot(r,xy)[0:2]
    wf.linear.z = bf.linear.z
    wf.angular.x = bf.angular.x
    wf.angular.y = bf.angular.y
    wf.angular.z = bf.angular.z
    return wf

def twist_wf_to_bf(wf):
    yaw = wf.angular.z
    yaw *= math.pi/180
    c = math.cos(yaw)
    s = math.sin(yaw)
    r_inv = np.array([[c, s, 0],[-s, c, 0],[0,0,1]])

    bf = Twist()
    xy = np.array([wf.linear.x, wf.linear.y, 1])
    bf.linear.x, bf.linear.y = np.dot(r,xy)[0:2]
    bf.linear.z = wf.linear.z
    bf.angular.x = wf.angular.x
    bf.angular.y = wf.angular.y
    bf.angular.z = wf.angular.z
    return wf
