#!/usr/bin/env python

"""
'Triangle'-button on controller to start the planner
'publish setpoints to '/set_point', Twist'
"""

import rospy
import numpy as np

from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool

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

global_state = S_INIT


#############
# Callbacks #
#############
def estimate_callback(data):
    global est_relative_position
    est_relative_position = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


# def initiate_mission_callback(data):
#     global global_state
#     global received_mission_time
#     global_state = S_PRE_MISSION
#     received_mission_time = rospy.get_time()


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
    # rospy.Subscriber('/initiate_mission', Empty, initiate_mission_callback)

    pub_take_off = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_land = rospy.Publisher("/ardrone/land", Empty, queue_size=10)

    pub_cv_switch = rospy.Publisher('/switch_on_off_cv', Bool, queue_size=10)

    pub_set_point = rospy.Publisher("/set_point", Twist, queue_size=1)
    set_point_msg = Twist()

    rospy.loginfo("Starting planner")

    mission_speed = 0.5 # m/s
    descend_speed = 0.1 # m/s
    publish_rate = 10 # Hz
    distance_margin = 0.2 # m
    margin = np.array([distance_margin]*3)
    land_margin = np.array([0.1, 0.1, 0.2])
    pre_mission_time = 3 # seconds

    hover_height = 2.0
    mission_delta_x = 1.5
    mission_delta_y = 3.0
    mission_height = 3.0

    hover_point = np.array([-0.1, 0.0, 2.0])
    land_point = np.array([-0.1, 0.0, 0.2])

    mission = np.array([
        [0.0                , 0.0               , hover_height  ],
        [0.0                , 0.0               , mission_height],
        [-mission_delta_x   , 0.0               , mission_height],
        [-mission_delta_x   , -mission_delta_y  , mission_height],
        [mission_delta_x    , -mission_delta_y  , mission_height],
        [mission_delta_x    , 0.0               , mission_height],
        [mission_delta_x    , mission_delta_y   , mission_height],
        [-mission_delta_x   , mission_delta_y   , mission_height],
        [-mission_delta_x   , 0.0               , mission_height],
        [0.0                , 0.0               , mission_height]
    ])

    # mission = np.array([
    #     [-0.1                , 0.0               , hover_height],
    #     [1.0                , 0.0               , hover_height],
    #     [-0.1                , 0.0               , hover_height]
    # ])


    ####################

    # S_INIT          = 0
    # S_ON_GROUND     = 1
    # S_TAKE_OFF      = 2
    # S_HOVER         = 3
    # S_PRE_MISSION   = 4
    # S_MISSION       = 5
    # S_RETURN_HOME   = 6
    # S_DESCEND       = 7
    # S_LAND          = 8

    
    rate = rospy.Rate(publish_rate) # Hz
    while not rospy.is_shutdown():
        use_cv = True

        current_position = est_relative_position

        if global_state == S_INIT:
            if current_position is not None:
                global_state = S_ON_GROUND

        elif global_state == S_ON_GROUND:
            pub_take_off.publish(Empty())
            timer_start = rospy.get_time()
            global_state = S_TAKE_OFF            

        elif global_state == S_TAKE_OFF:
            curr_time = rospy.get_time()
            if curr_time - timer_start < 1:
                pub_take_off.publish(Empty())
            else:
                publish_set_point(pub_set_point, hover_point)

            if is_position_close_to_goal(current_position, hover_point, margin):
                timer_start = rospy.get_time()
                global_state = S_HOVER

        elif global_state == S_HOVER:
            curr_time = rospy.get_time()
            if curr_time - timer_start > 3:
                global_state = S_PRE_MISSION
                received_mission_time = rospy.get_time()

        elif global_state == S_PRE_MISSION:
            use_cv = False
            curr_time = rospy.get_time()

            if curr_time - received_mission_time > pre_mission_time:
                mission_count = 0

                prev_major_set_point = mission[0]
                next_major_set_point = mission[1]
                next_minor_set_point = next_major_set_point

                global_state = S_MISSION

        elif global_state == S_MISSION:
            use_cv = False
            # Time to change to next major setpoint
            if get_distance(next_minor_set_point, next_major_set_point) < distance_margin:
                if mission_count == len(mission)-1:
                    global_state = S_RETURN_HOME
                    publish_set_point(pub_set_point, hover_point)
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
                next_minor_set_point += step_distance
                publish_set_point(pub_set_point, next_minor_set_point)

        elif global_state == S_RETURN_HOME:
            if is_position_close_to_goal(current_position, hover_point, margin):
                next_minor_set_point = hover_point
                next_major_set_point = land_point
                
                translation = next_major_set_point - next_minor_set_point
                distance = np.linalg.norm(translation)

                step_time = distance / descend_speed
                num_steps = step_time * publish_rate
                descend_count = 0
                step_distance = translation / num_steps
                global_state = S_DESCEND
            else:
                publish_set_point(pub_set_point, hover_point)
                
        elif global_state == S_DESCEND:
            if is_position_close_to_goal(current_position, land_point, land_margin):
                pub_land.publish(Empty())
                global_state = S_LAND
            else:
                if descend_count < num_steps:
                    next_minor_set_point += step_distance
                    descend_count += 1
                publish_set_point(pub_set_point, next_minor_set_point)

        elif global_state == S_LAND:
            publish_set_point(pub_set_point, np.zeros(3))
            rospy.loginfo("Autonomy disabled")
            break
        else:
            rospy.loginfo("Invalid state")

        pub_cv_switch.publish(Bool(use_cv))
    
        print_state(global_state)

        rate.sleep()
    
    
if __name__ == '__main__':
    main()