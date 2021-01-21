#!/usr/bin/env python

"""
'Triangle'-button on controller to start the mission
'publish setpoints to '/set_point', Twist'
"""

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool

import time

est_relative_position = None

# States of the quadcopter
S_INIT          = 0
S_ON_GROUND     = 1
S_TAKE_OFF      = 2
S_HOVER         = 3
S_PRE_MISSION   = 4
S_MISSION       = 5
S_RETURN_HOME   = 6
S_DESCEND       = 7
S_LAND          = 8

STATE_TEXT = [
    "INIT",
    "ON GROUND",
    "TAKE OFF",
    "HOVER",
    "PRE MISSION",
    "MISSION",
    "RETURN HOME",
    "DESCEND",
    "LAND"
]

global_state = S_HOVER


#############
# Callbacks #
#############
def estimate_callback(data):
    global est_relative_position
    est_relative_position = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


def initiate_mission_callback(data):
    global global_state
    global received_mission_time
    global_state = S_PRE_MISSION
    received_mission_time = rospy.get_time()


#######################################

def print_state(state):
    rospy.loginfo("State: " + STATE_TEXT[state])


def get_distance(point_a, point_b):
    translation = point_b - point_a
    distance = np.linalg.norm(translation)
    return distance


def is_position_close_to_goal(curr_position, goal, margin):
    # rospy.loginfo(str(np.abs(curr_position[:3] - goal)))
    return np.all(np.abs(curr_position[:3] - goal) < margin)


def publish_set_point(pub_set_point, set_point):
    set_point_msg = Twist()
    set_point_msg.linear.x = set_point[0]
    set_point_msg.linear.y = set_point[1]
    set_point_msg.linear.z = set_point[2]
    pub_set_point.publish(set_point_msg)


def main():
    global global_state
    rospy.init_node('planner', anonymous=True)

    rospy.Subscriber('/estimate/dead_reckoning', Twist, estimate_callback)
    rospy.Subscriber('/initiate_mission', Empty, initiate_mission_callback)

    pub_set_point = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_land = rospy.Publisher("/ardrone/land", Empty, queue_size=10)

    set_point_msg = Twist()

    rospy.loginfo("Starting mission module")

    mission_speed = 0.4 # m/s
    publish_rate = 10 # Hz
    distance_margin = 0.01 # m
    distance_speed_reduction_margin = 1.0 # m

    margin = np.array([distance_margin]*3)
    land_margin = np.array([0.1, 0.1, 0.2])
    pre_mission_time = 1 # second(s)

    hover_height = 1.5
    mission_height = 5.0

    hover_point = np.array([0.0, 0.0, hover_height])

    # Experiment A
    # mission = np.array([
    #     [-0.1                , 0.0               , hover_height  ],
    #     [-0.1                , 0.0               , mission_height],
    #     [-0.1                , 0.0               , hover_height  ]
    # ])

    # Experiment C
    # mission = np.array([
    #     [0.0, 0.0, 1.0],
    #     [0.0, 0.0, 3.0]
    # ])

    # Experiment D
    # mission = np.array([
    #     [0.0, 0.0, 1.5],
    #     [3.0, 0.0, 1.5],
    #     [0.0, 0.0, 1.5]
    # ])

    # Experiment E
    mission = np.array([
        [1.0, 2.0, 4.0],
        [0.0, 0.0, 2.0],
        [0.0, 0.0, 0.2]
    ])

    land_after_mission = True

        
    rate = rospy.Rate(publish_rate) # Hz
    while not rospy.is_shutdown():
        use_cv = True

        current_position = est_relative_position

        if global_state == S_INIT:
            global_state = S_HOVER

        elif global_state == S_ON_GROUND:
            global_state = S_HOVER           

        elif global_state == S_TAKE_OFF:
            global_state = S_HOVER

        elif global_state == S_HOVER:
            global_state = S_HOVER

        elif global_state == S_PRE_MISSION:
            curr_time = rospy.get_time()

            if curr_time - received_mission_time > pre_mission_time:
                mission_count = 0

                prev_major_set_point = mission[0]
                next_major_set_point = mission[1]
                next_minor_set_point = next_major_set_point

                global_state = S_MISSION

        elif global_state == S_MISSION:
            # Time to change to next major setpoint
            distance_to_target = get_distance(next_minor_set_point, next_major_set_point)

            if distance_to_target < distance_margin:
                if mission_count == len(mission)-1:
                    mission_count = 0
                    if land_after_mission:
                        pub_land.publish(Empty())
                        global_state = S_LAND
                    else:
                        global_state = S_HOVER
                        publish_set_point(pub_set_point, hover_point)
                        break
                else:
                    next_major_set_point = mission[mission_count+1]

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

        elif global_state == S_RETURN_HOME:
            global_state = S_HOVER
                
        elif global_state == S_DESCEND:
            global_state = S_HOVER

        elif global_state == S_LAND:
            publish_set_point(pub_set_point, np.zeros(3))
            rospy.loginfo("Autonomy disabled")
            break
    
        print_state(global_state)

        rate.sleep()
    
    
if __name__ == '__main__':
    main()