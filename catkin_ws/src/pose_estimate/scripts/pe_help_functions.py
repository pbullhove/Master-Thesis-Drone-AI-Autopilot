#!/usr/bin/env python
import math
import rospy
import numpy as np
from geometry_msgs.msg import Twist


def deg2rad(deg):
    return deg * 3.141592 / 180.0

def twist_to_array(twist):
    """ Transforms a pose Twist into 6d array of pose. """
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr

def to_Twist(array):
    """ Transforms a 6d array of pose into a pose Twist """
    tw = Twist()
    tw.linear.x = array[0]
    tw.linear.y = array[1]
    tw.linear.z = array[2]
    tw.angular.x = array[3]
    tw.angular.y = array[4]
    tw.angular.z = array[5]
    return tw

def twist_to_array(twist):
    """ Transforms a pose Twist into 6d array of pose. """
    arr = np.array([twist.linear.x, twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z])
    return arr


def wf_to_bf(wf,yaw):
    """
    Transforms world frame pose into body frame pose. This is by performing a -yaw rotation about the
    z-axis of the x and y coordinates. z, pitch, roll, yaw unchanged.
    input:
        wf: array[6] float - pose in world frame
        yaw: float - Degrees. Best estimated yaw angle for transformation.

    output:
        bf: array[6] float - pose in body frame.
    """
    yaw *= math.pi/180
    c = math.cos(yaw)
    s = math.sin(yaw)
    r_inv = np.array([[c, s, 0],[-s, c, 0],[0,0,1]])

    wf_xy1 = np.array([wf[0], wf[1], 1])
    bf_xy1 = np.dot(r_inv,wf_xy1)
    bf = np.array([bf_xy1[0], bf_xy1[1], wf[2], wf[3], wf[4], wf[5]])
    return bf


def angleFromTo(ang, min, max):
    """
    Angle, min, max: Degrees
    Moves ang into the desired range of coordinates.
    Mostly used for yaw -> [-180, 180]
    """
    if ang < min:
        ang += 360
    if ang > max:
        ang -= 360
    return ang



def bf_to_wf(bf,yaw=None):
    """
    Transforms body frame pose into world frame pose. This is by performing a yaw rotation about the
    z-axis of the x and y coordinates. z, pitch, roll, yaw unchanged.
    input:
        bf: array[6] float - pose in body frame
        yaw: float - Degrees. Best estimated yaw angle for transformation.

    output:
        wf: array[6] float - pose in world frame.
    """
    y = yaw if yaw is not None else bf[5]
    y *= math.pi/180
    c = math.cos(y)
    s = math.sin(y)
    r = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])

    bf_xy1 = np.array([bf[0], bf[1], 1])
    wf_xy1 = np.dot(r,bf_xy1)

    wf = np.array([wf_xy1[0], wf_xy1[1], bf[2], bf[3], bf[4], bf[5]])
    return wf


def twist_bf_to_wf(bf, yaw=None):
    """
    Transforms body frame pose into world frame pose. This is by performing a yaw rotation about the
    z-axis of the x and y coordinates. z, pitch, roll, yaw unchanged.
    input:
        bf: Twist() - pose in body frame
        yaw: float - Degrees. Best estimated yaw angle for transformation.

    output:
        wf: Twist - pose in world frame.
    """
    y = yaw if yaw is not None else bf.angular.z
    y *= math.pi/180
    c = math.cos(y)
    s = math.sin(y)
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
    """
    Transforms world frame pose into world frame pose. This is by performing a -yaw rotation about the
    z-axis of the x and y coordinates. z, pitch, roll, yaw unchanged.
    input:
        bf: Twist() - pose in world frame
        yaw: float - Degrees. Best estimated yaw angle for transformation.

    output:
        wf: Twist - pose in body frame.
    """
    y = yaw if yaw is not None else wf.angular.z
    y *= math.pi/180
    c = math.cos(y)
    s = math.sin(y)
    r_inv = np.array([[c, s, 0],[-s, c, 0],[0,0,1]])

    bf = Twist()
    xy = np.array([wf.linear.x, wf.linear.y, 1])
    bf.linear.x, bf.linear.y = np.dot(r,xy)[0:2]
    bf.linear.z = wf.linear.z
    bf.angular.x = wf.angular.x
    bf.angular.y = wf.angular.y
    bf.angular.z = wf.angular.z
    return wf
