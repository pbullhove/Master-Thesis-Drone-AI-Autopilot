#!/usr/bin/env python
import rospy
import numpy as np
import config as cfg
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool
import copy as cp
import math
from timer import Timer, TimerError

# Quadcopter States
STATE_INIT = "STATE_INIT"
STATE_TAKEOFF = "STATE_TAKEOFF"
STATE_HOVER = "STATE_HOVER"
STATE_LANDING = "STATE_LANDING"
STATE_PHOTOTWIRL = "STATE_PHOTOTWIRL"
STATE_MOVING = "STATE_MOVING"
STATE_ERROR = "STATE_ERROR"
STATE_IDLE = "STATE_IDLE"

empty = Empty()
desired_pose = Twist()
desired_pose.linear.z = cfg.takeoff_height
hover_pose = cp.deepcopy(desired_pose)
current_pose = Twist()
hover_timer = Timer()
error_timer = Timer()
state = STATE_INIT
landing_complete = False
phototwirl_complete = False
photos_taken = 0
received_estimate = False



def bf_to_wf(bf):
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


#############
# Callbacks #
#############


def estimate_callback(data):
    global current_pose
    global received_estimate
    #print('received estimate')
    received_estimate = True
    bf_pose = data
    current_pose = bf_to_wf(bf_pose)

def landing_complete_callback(data):
    global landing_complete
    print('received landing complete')
    landing_complete = True


def angular_distance(or_a, or_b):
    return abs(or_a.x + or_a.y + or_a.z - or_b.x - or_b.y - or_b.z)

def euclidean_distance(pos_a, pos_b):
    a = np.array([pos_a.x, pos_a.y, pos_a.z])
    b = np.array([pos_b.x, pos_b.y, pos_b.z])
    return np.linalg.norm(a - b)

def close_enough(pose_a, pose_b):
    pos_a = pose_a.linear
    pos_b = pose_b.linear
    or_a = pose_a.angular
    or_b = pose_b.angular

    ang = angular_distance(or_a, or_b)
    euc = euclidean_distance(pos_a, pos_b)

    return ang < cfg.close_enough_ang and euc < cfg.close_enough_euc



def main():
    rospy.init_node('complete_mission',anonymous=True)

    global received_estimate
    global landing_complete
    global desired_pose
    global hover_pose
    global phototwirl_complete
    global photos_taken
    global empty
    global state
    pub_desired_pose = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_start_takeoff = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_start_automated_landing = rospy.Publisher('/initiate_automated_landing', Empty, queue_size=1)
    pub_save_front_camera_photo = rospy.Publisher('/take_still_photo_front',Empty, queue_size=1)
    control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    pid_off = rospy.Publisher('/pid_on_off', Bool, queue_size=1)
    rospy.Subscriber('/filtered_estimate', Twist, estimate_callback)

    rospy.Subscriber('/ardrone/land', Empty, landing_complete_callback)

    def init_complete():
        a = pub_desired_pose.get_num_connections() > 0
        b = pub_start_takeoff.get_num_connections() > 0
        c = pub_start_automated_landing.get_num_connections() > 0
        d = pub_save_front_camera_photo.get_num_connections() > 0
        e = control_pub.get_num_connections() > 0
        return all([a, b, c, d, e, received_estimate])

    takeoff_timer = Timer()
    print('State is:', state)
    start_pose = None
    while not rospy.is_shutdown():
        if state not in [STATE_ERROR, STATE_IDLE, STATE_INIT] and error_timer.is_timeout():
            state = STATE_ERROR

        if state == STATE_INIT:
            if init_complete():
                error_timer.start(cfg.error_timeout_time)
                state = STATE_TAKEOFF
                print('switching to: ', state)
                start_pose = cp.deepcopy(current_pose)
                takeoff_timer.start(3)
                pub_start_takeoff.publish(empty)
                pub_desired_pose.publish(desired_pose)

        if state == STATE_TAKEOFF:
            if close_enough(current_pose, desired_pose):
                error_timer.reset()
                state = STATE_MOVING
                desired_pose.linear.x = 10
                pub_desired_pose.publish(desired_pose)
                print('switching to: ', state)
            elif takeoff_timer.is_timeout() and close_enough(current_pose, start_pose):
                pub_start_takeoff.publish(empty)
                pub_desired_pose(desired_pose)
                print('publishing takeoff again!')
                takeoff_timer.reset()

        if state == STATE_HOVER:
            if close_enough(current_pose, desired_pose):
                try:
                    hover_timer.start(5)
                except TimerError as t: #timer already running
                    if hover_timer.is_timeout():
                        hover_timer.stop()
                        if phototwirl_complete:
                            error_timer.reset()
                            state = STATE_LANDING
                            print('switching to: ', state)
                            pub_start_automated_landing.publish(empty)
                        else:
                            error_timer.reset()
                            state = STATE_PHOTOTWIRL
                            print('switching to: ', state)

        if state == STATE_MOVING:
            if close_enough(current_pose, desired_pose):
                error_timer.reset()
                state = STATE_HOVER
                print('switching to: ', state)

        if state == STATE_PHOTOTWIRL:
            if close_enough(current_pose, desired_pose):
                print('photo: close enough', current_pose, desired_pose)
                if photos_taken < 4:
                    pub_save_front_camera_photo.publish(empty)
                    photos_taken += 1
                    #desired_pose.angular.z = [135, 45, -45, -135][photos_taken]
                    desired_pose.angular.z = [0, 90, 179, -90, 0][photos_taken]
                    pub_desired_pose.publish(desired_pose)
                else:
                    state = STATE_MOVING
                    error_timer.reset()
                    print('switching to: ', state)
                    phototwirl_complete = True
                    desired_pose = cp.deepcopy(hover_pose)
                    pub_desired_pose.publish(desired_pose)

        if state == STATE_LANDING:
            if landing_complete:
                error_timer.stop()
                state = STATE_IDLE
                print('switching to: ', state)

        if state == STATE_ERROR:
            print('switching to: ', state)
            msg = Twist()
            off = Bool()
            off.data = False
            msg.linear.z = -cfg.error_descent_vel
            while True:
                pid_off.publish(off)
                control_pub.publish(msg)

if __name__ == "__main__":
    main()
