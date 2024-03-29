#!/usr/bin/env python
"""
Mission control center.
State machine approach for quadcopter mission control.

Mission defined by:
    mission_plan -  array of action strings. Quadcopter will perform actions in sequence.

    available commands:
        INIT: wait for init_compete(), then start
        TAKEOFF: perform quadcopter takeoff
        MOVE to [x,y,z,0,0,yaw]
        PHOTOTWIRL: 360 degree spin and take photos in all directions.
        HOVER: Hover at desired_pose for cfg.hover_duration seconds
        LANDING: Perform automated landing over landing platform.
        IDLE: do nothing

Subscribes to:
    /filtered_estimate: Twist - Current pose estimate
    /ardrone/land: Emptty - For knowing that landing completed successfully.

Publishes to:
    /set_point: Twist - For quadcopter navigation, sendt to PID controller to complete.
    /ardrone/takeoff: Empty - For quadcopter takeoff.
    /initiate_automated_landing: Empty - For initiating automated landing.
    /take_still_photo_front: Empty - Front photo capture
    /take_still_photo_bottom: Empty - Bottom photo capture
    /cmd_vel: Twist - For emergency state: Direct control of quadcopter:
    /pid_on_off: Bool - For emergency state: Direct control of quadcopter, toggle pid off.

"""


import rospy
import numpy as np
import sys
import config as cfg
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool
import copy as cp
import math
from timer import Timer, TimerError

import sys
sys.path.append('../utilities')
import help_functions as hlp

# Quadcopter States
INIT = "INIT"
TAKEOFF = "TAKEOFF"
HOVER = "HOVER"
LANDING = "LANDING"
PHOTOTWIRL = "PHOTOTWIRL"
MOVING = "MOVING"
ERROR = "ERROR"
IDLE = "IDLE"

empty = Empty()
takeoff_pose = Twist()
takeoff_pose.linear.z = cfg.takeoff_height
desired_pose = Twist()
current_pose = Twist()
start_pose = Twist()
error_timer = Timer()
timer = Timer()
state = INIT
photos_taken = 0
received_estimate = False
landing_complete = False
mission_step = 0



args = sys.argv
print(args)
if len(args) == 1:
    mission_plan = ["INIT", "TAKEOFF", "HOVER", "MOVE TO [5,0,3,0,0,0]", "HOVER", "PHOTOTWIRL", "HOVER", "MOVE TO [0,0,3,0,0,0]", "HOVER", "LANDING", "IDLE"]
else:
    if args[1].lower() == "thl":
        mission_plan = ["INIT", "TAKEOFF","HOVER", "LANDING"]
    elif args[1].lower() == "land":
        mission_plan = ["INIT","MOVE TO [0,0,3,0,0,0]", "MOVE TO [0,0,1.5,0,0,0]", "LANDING"]
    elif args[1].lower() == "thhhl":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "HOVER", "LANDING", "IDLE"]
    elif args[1].lower() == "movefar":
        mission_plan = ["INIT", "TAKEOFF", "MOVE TO [0,0,30,0,0,0]", "HOVER", "MOVE TO [0,0,3,0,0,0]", "HOVER", "LANDING"]
    elif args[1].lower() == "photomission_here":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "PHOTOTWIRL", "MOVE TO [0,0,2,0,0,0]", "HOVER", "LANDING", "IDLE"]
    elif args[1].lower() == "photomission_there":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "HOVER", "MOVE TO [10,0,5,0,0,0]", "PHOTOTWIRL", "MOVE TO [10,10,5,0,0,0]", "MOVE TO [0,10,5,0,0,0]", "MOVE TO [0,0,5,0,0,0]","HOVER",  "LANDING", "IDLE"]
    elif args[1].lower() == "moveabit":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "MOVE TO [1,1,2,0,0,0]", "HOVER", "MOVE TO [0,0,2,0,0,0]", "HOVER", "LANDING", "IDLE"]
    elif args[1].lower() == "move":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "MOVE TO [5,5,2,0,0,0]", "HOVER", "MOVE TO [0,0,2,0,0,0]", "HOVER", "LANDING", "IDLE"]
    elif args[1].lower() == "hex":
        mission_plan = ["INIT", "TAKEOFF", "HOVER", "MOVE TO [10,0,5,0,0,0]", "MOVE TO [5,9,5,0,0,0]", "MOVE TO [-5,9,5,0,0,0]","MOVE TO [-10,0,5,0,0,0]","MOVE TO [-5,-9,5,0,0,0]", "MOVE TO [5,-9,5,0,0,0]", "MOVE TO [10,0,5,0,0,0]", "MOVE TO [0,0,5,0,0,0]", "HOVER", "LANDING", "IDLE"]
    else:
        raise("unknown mission")


#############
# Callbacks #
#############


def estimate_callback(data):
    """ Receives best current quadcopter pose estimate. """
    global current_pose
    global received_estimate
    #print('received estimate')
    received_estimate = True
    bf_pose = data
    current_pose = hlp.twist_bf_to_wf(bf_pose)

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
    rospy.init_node('mission',anonymous=True)

    global received_estimate
    global landing_complete
    global desired_pose
    global phototwirl_complete
    global photos_taken
    global empty
    global state
    global mission_step
    global timer
    pub_desired_pose = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_start_takeoff = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_start_automated_landing = rospy.Publisher('/initiate_automated_landing', Empty, queue_size=1)
    pub_save_front_camera_photo = rospy.Publisher('/take_still_photo_front',Empty, queue_size=1)
    control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    toggle_pid = rospy.Publisher('/pid_on_off', Bool, queue_size=1)
    rospy.Subscriber('/filtered_estimate', Twist, estimate_callback)

    rospy.Subscriber('/ardrone/land', Empty, landing_complete_callback)

    def init_complete():
        """ Init is complete if all modules that are to communicate with mission are active,
        as well as pose estimates are arriving.
        """
        a = pub_desired_pose.get_num_connections() > 0
        b = pub_start_takeoff.get_num_connections() > 0
        c = pub_start_automated_landing.get_num_connections() > 0
        d = pub_save_front_camera_photo.get_num_connections() > 0
        e = control_pub.get_num_connections() > 0
        return all([a, b, c, d, e, received_estimate])


    def transition_to(new_state):
        """
        Handles mission state transitions.
        Changes state to new state, as well as performing required actions to initiate new state.
        """
        global desired_pose
        global state
        global timer
        global start_pose
        global photos_taken

        error_timer.reset()
        state = new_state
        print('Switching to state: ', state)
        if state == 'IDLE':
            landing_complete = False
        elif state == "ERROR":
            pass
        elif state == "INIT":
            landing_complete = False
            pass
        elif state == "TAKEOFF":
            if not cfg.is_simulator:
                off = Bool()
                off.data = False
                toggle_pid.publish(off)
            start_pose = cp.deepcopy(current_pose)
            desired_pose = cp.deepcopy(takeoff_pose)
            timer.start(cfg.takeoff_timer_duration)
            pub_start_takeoff.publish(empty)
            pub_desired_pose.publish(desired_pose)
        elif state == "HOVER":
            timer.start(cfg.hover_duration)
        elif state.startswith('MOVE TO') or state.startswith('MOVING'):
            try:
                setpoint_str = new_state.split('[')[-1].split(']')[0].split(',')
                setpoint = [float(i) for i in setpoint_str]
                desired_pose = hlp.to_Twist(setpoint)
            except Exception as e:
                print('faulty move statement. Moving to platform home.')
                desired_pose = hlp.to_Twist([0,0,1,0,0,0])
            finally:
                state = "MOVING"
                pub_desired_pose.publish(desired_pose)
        elif state == "PHOTOTWIRL":
            photos_taken = 0
        elif state == "LANDING":
            start_pose = cp.deepcopy(current_pose)
            timer.start(cfg.landing_timer_duration)
            pub_start_automated_landing.publish(empty)
        else:
            raise TypeError("State name undefined: No such state exists. ")


    print('State is:', state)
    error_timer.start(cfg.error_timer_duration)
    while not rospy.is_shutdown():
        """
        Transition checking loop.
        Constantly checks for state completion requirements. Transitions to next step in mission_plan if finished.
        Will trainsitions are handled by the function transition_to(new_state).
        """
        if state not in [ERROR, IDLE, INIT] and error_timer.is_timeout():
            print('error timer - timeOut!')
            transition_to(ERROR)

        elif state == INIT:
            if init_complete():
                mission_step += 1
                transition_to(mission_plan[mission_step])

        elif state == TAKEOFF:
            if close_enough(current_pose, desired_pose):
                timer.stop()
                mission_step += 1
                if not cfg.is_simulator:
                    pid_on = Bool()
                    pid_on.data = True
                    toggle_pid.publish(pid_on)
                transition_to(mission_plan[mission_step])
            elif timer.is_timeout() and close_enough(current_pose, start_pose):
                timer.stop()
                transition_to("TAKEOFF")

        elif state == HOVER:
            if close_enough(current_pose, desired_pose) and timer.is_timeout():
                timer.stop()
                mission_step += 1
                transition_to(mission_plan[mission_step])

        elif state == MOVING:
            if close_enough(current_pose, desired_pose):
                mission_step += 1
                transition_to(mission_plan[mission_step])

        elif state == PHOTOTWIRL:
            if close_enough(current_pose, desired_pose):
                #print('photo: close enough', current_pose, desired_pose)
                if photos_taken < 4:
                    pub_save_front_camera_photo.publish(empty)
                    print('Captured photo at yaw: ', current_pose.angular.z)
                    photos_taken += 1
                    desired_pose.angular.z = [0, 90, 179, -90, 0][photos_taken]
                    pub_desired_pose.publish(desired_pose)
                else:
                    mission_step += 1
                    transition_to(mission_plan[mission_step])

        elif state == LANDING:
            if landing_complete:
                mission_step += 1
                try:
                    transition_to(mission_plan[mission_step])
                except IndexError as e:
                    transition_to("IDLE")
            elif timer.is_timeout() and close_enough(current_pose, start_pose):
                timer.stop()
                transition_to("LANDING")

        elif state == ERROR:
            msg = Twist()
            off = Bool()
            off.data = False
            msg.linear.z = cfg.error_descent_vel
            while True:
                toggle_pid.publish(off)
                control_pub.publish(msg)

if __name__ == "__main__":
    main()
