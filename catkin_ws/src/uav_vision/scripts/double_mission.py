#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool
import copy as cp

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
desired_pose.linear.z = 3
hover_pose = cp.deepcopy(desired_pose)
current_pose = Twist()
hover_timer = Timer()
error_timer = Timer()
state = STATE_INIT
landing_complete = False
phototwirl_complete = False
photos_taken = 0
received_estimate = False



#############
# Callbacks #
#############
def estimate_callback(data):
    global current_pose
    global received_estimate
    #print('received estimate')
    received_estimate = True
    current_pose = data

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

    return ang < 5 and euc < 0.1



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
    rospy.Subscriber('/estimate/dead_reckoning', Twist, estimate_callback)
    rospy.Subscriber('/ardrone/land', Empty, landing_complete_callback)

    def init_complete():
        a = pub_desired_pose.get_num_connections() > 0
        b = pub_start_takeoff.get_num_connections() > 0
        c = pub_start_automated_landing.get_num_connections() > 0
        d = pub_save_front_camera_photo.get_num_connections() > 0
        e = control_pub.get_num_connections() > 0
        return all([a, b, c, d, e, received_estimate])

    print('State is:', state)
    while not rospy.is_shutdown():
        if state not in [STATE_ERROR, STATE_IDLE, STATE_INIT] and error_timer.is_timeout():
            state = STATE_ERROR

        if state == STATE_INIT:
            if init_complete():
                error_timer.start(100000)
                error_timer.reset()
                state = STATE_TAKEOFF
                print('switching to: ', state)
                pub_start_takeoff.publish(empty)
                pub_desired_pose.publish(desired_pose)

        if state == STATE_TAKEOFF:
            if close_enough(current_pose, desired_pose):
                error_timer.reset()
                print('ye close enough')
                state = STATE_MOVING
                desired_pose.linear.x = 3
                pub_desired_pose.publish(desired_pose)
                print('switching to: ', state)

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
            msg.linear.z = -0.3
            while True:
                pid_off.publish(off)
                control_pub.publish(msg)

if __name__ == "__main__":
    main()
