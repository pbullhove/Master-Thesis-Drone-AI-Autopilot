#!/usr/bin/env python

"""
Automated landing module for performing a safe automated landing with the Parrot AR2.0 Quadcopter.
Is a state machine implementation. Start this in close proximity to the landing platform.

Subscribes to:
    /filtered_estimate: Twist - current estimated pose.
    /initiate_automated_landing: Empty - for starting automated landing.
Publishes to:
    /set_point: Twist - for communicating setpoints to the PID controller
    /ardrone/land: Empty - for communicating to the ardrone to drop down and cut motors.

When using PS4 controller:
    'Triangle'-button on controller to start the mission
"""

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool

import time

est_pose = None

# States of the quadcopter
S_INIT          = "INIT"
S_PRE_LANDING   = "PRE LANDING"
S_LANDING       = "LANDING"
S_LANDED        = "LANDED"

global_state = S_INIT

global_mission = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.2]
])



#############
# Callbacks #
#############
def estimate_callback(data):
    global est_pose
    est_pose = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


def initiate_landing_callback(data):
    global received_start_time
    global position_at_start_time
    global global_mission
    global global_state

    received_start_time = rospy.get_time()
    position_at_start_time = est_pose
    global_mission[0] = position_at_start_time[:3]
    global_state = S_PRE_LANDING

#######################################

def print_state(state):
    rospy.loginfo("State: " + state)


def get_distance(point_a, point_b):
    translation = point_b - point_a
    distance = np.linalg.norm(translation)
    return distance


def is_position_close_to_goal(curr_position, goal, margin):
    return np.all(np.abs(curr_position[:3] - goal) < margin)


def publish_set_point(pub_set_point, set_point):
    set_point_msg = Twist()
    set_point_msg.linear.x = set_point[0]
    set_point_msg.linear.y = set_point[1]
    set_point_msg.linear.z = set_point[2]
    pub_set_point.publish(set_point_msg)


def main():
    global global_state
    rospy.init_node('automated_landing', anonymous=True)

    rospy.Subscriber('/filtered_estimate', Twist, estimate_callback)
    rospy.Subscriber('/initiate_automated_landing', Empty, initiate_landing_callback)
    pub_set_point = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_land = rospy.Publisher("/ardrone/land", Empty, queue_size=10)

    set_point_msg = Twist()
    rospy.loginfo("Starting automated landing module")
    print_state(global_state)

    publish_rate = 10 # Hz
    mission_speed = 0.4 # m/s
    distance_margin = 0.01 # m
    distance_speed_reduction_margin = 1.0 # m

    margin = np.array([distance_margin]*3)
    pre_mission_time = 1 # second(s)

    rate = rospy.Rate(publish_rate) # Hz
    while not rospy.is_shutdown():
        use_cv = True

        current_position = est_pose

        if global_state == S_INIT:
            pass

        elif global_state == S_PRE_LANDING:
            curr_time = rospy.get_time()

            if curr_time - received_start_time > pre_mission_time:
                mission_count = 0

                prev_major_set_point = global_mission[0]
                next_major_set_point = global_mission[1]
                next_minor_set_point = next_major_set_point

                global_state = S_LANDING
                print_state(global_state)

        elif global_state == S_LANDING:
            # Time to change to next major setpoint
            distance_to_target = get_distance(next_minor_set_point, next_major_set_point)

            if distance_to_target < distance_margin:
                if mission_count == len(global_mission)-1:
                    mission_count = 0
                    pub_land.publish(Empty())
                    global_state = S_LANDED
                    print_state(global_state)
                else:
                    next_major_set_point = global_mission[mission_count+1]

                    translation = next_major_set_point - prev_major_set_point
                    distance = np.linalg.norm(translation)

                    step_time = distance / mission_speed
                    num_steps = step_time * publish_rate
                    step_distance = translation / num_steps
                    next_minor_set_point = prev_major_set_point

                    prev_major_set_point = next_major_set_point
                    publish_set_point(pub_set_point, next_minor_set_point)

                    mission_count += 1
            else:

                if distance_to_target < distance_speed_reduction_margin:
                    speed_reduction = np.maximum(distance_to_target / distance_speed_reduction_margin, 0.1)
                else:
                    speed_reduction = 1.0

                next_minor_set_point += step_distance*speed_reduction
                publish_set_point(pub_set_point, next_minor_set_point)


        elif global_state == S_LANDED:
            publish_set_point(pub_set_point, np.zeros(3))
            rospy.loginfo("Landed: Ready to go again!")
            global_state = S_INIT
            print_state(global_state)


        rate.sleep()


if __name__ == '__main__':
    main()
