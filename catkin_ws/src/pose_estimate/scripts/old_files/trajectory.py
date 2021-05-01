#!/usr/bin/env python

import rospy

from std_msgs.msg import Empty
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
import numpy as np
from scipy.misc import imsave
import os
from scipy.spatial.transform import Rotation as R

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import math


HELIPAD_POS_X = 1.0
HELIPAD_POS_Y = 1.0
HELIPAD_POS_Z = 0.54

# set_points = np.array([
#     [0.0, 0.0, 1.0],
#     [-1.0, -1.0, 1.0],
#     [1.0, -1.0, 1.0],
#     [1.0, 1.0, 1.0],
#     [-1.0, 1.0, 1.0],
#     [0.0, 0.0, 1.0]
# ])

# Dimentions
#         width        
# *                    *
# *                    *
# *                    * length
# *                    *
# *                    *
# *                    *

# ODD index round (starting at round 1)
# 0 5 l/10             *
# *                    0 3 l/10
# 0 1 l/10             *
# *                    0 -1 l/10
# 0 -3 l /10           *
# *                    0 -5 l/10

# EVEN index round
# *                    0 5 l/10
# 0 3 l/10             *
# *                    0 1 l/10
# 0 -1 l/10            *
# *                    0 -3 l /10
# 0 -5 l/10            *

speed = 0.2 # m/s

min_height = 1.0
max_height = 12.0
height_step = 0.5

# Ascending 1 meter with speed 0.1 and step 0.1 takes approximately 709 s

# Speed 100: 3.58488573363
# Speed 0.2: 1792s: 29 min


# 1792 s at 14 runder: 128 s at hver runde
# # 2.5 oscilasjoner at hver runde: 51.2 s at hver oscilasjon

# speed = 0.2 # m/s

# min_height = 1.0
# max_height = 8.0
# height_step = 0.5


# Speed 100: 9.87s
# Speed 10: 39.57s
# Speed 5: 75s
# Speed 0.2: 1875s

# Settings for speed = 0.5
# At 1m
# length = 0.1 # in UAV's x direction
# width = 0.7 # in UAV's y direction

# At 3m
# length = 1.0 # in UAV's x direction
# width = 2.8 # in UAV's y direction

# At 5m
# length = 1.9 # in UAV's x direction
# width = 4.9 # in UAV's y direction

# At 7m
# length = 2.8 # in UAV's x direction
# width = 7.0 # in UAV's y direction


# sign = 1 or sign = -1
# set_points_element = np.array([
#     [0.0        , 0.0 , height],
#     [sign*5*l_10, -w_2, height],
#     [sign*3*l_10,  w_2, height],
#     [sign*1*l_10, -w_2, height],
#     [-sign*1*l_10,  w_2, height],
#     [-sign*3*l_10, -w_2, height],
#     [-sign*5*l_10,  w_2, height],
# ])


# Add all the setpoints to an array, starting at the mid bottom
set_points = np.array([[0,0,min_height]])

height = min_height
sign = 1
while height < max_height:

    l_slope = 0.85
    l_inter = -0.80

    w_slope = 1.45
    w_inter = -0.80

    # Must be multiplied with 2:
    # ('slope l:',        0.46135304
    # ('intercept l:',    -0.3464550733967431
    # ('slope w:',        0.78080833
    # ('intercept w:',    -0.36154347233387263

    # l = max(-0.35 + 0.45*height, 0)
    # w = max(-0.35 + 1.05*height, 0)

    # l_10 = l/10.0
    # w_2 = w/2.0

    # Set up flight pattern (description in master thesis)
    # * Avoiding stand still value for all parameters (x,y,z) to avoid bias of this value

    # h_0 = height + height_step*0.0
    # h_1 = height + height_step*0.14
    # h_2 = height + height_step*0.29
    # h_3 = height + height_step*0.43
    # h_4 = height + height_step*0.57
    # h_5 = height + height_step*0.71
    # h_6 = height + height_step*0.86

    h_1 = height + height_step*0.16
    h_2 = height + height_step*0.32
    h_3 = height + height_step*0.48
    h_4 = height + height_step*0.64
    h_5 = height + height_step*0.8
    h_6 = height + height_step*1.0

    l_1 = sign*5*max(l_inter + l_slope*h_1, 0)/10.0
    l_2 = sign*3*max(l_inter + l_slope*h_2, 0)/10.0
    l_3 = sign*1*max(l_inter + l_slope*h_3, 0)/10.0
    l_4 = -sign*1*max(l_inter + l_slope*h_4, 0)/10.0
    l_5 = -sign*3*max(l_inter + l_slope*h_5, 0)/10.0
    l_6 = -sign*5*max(l_inter + l_slope*h_6, 0)/10.0

    w_1 = -max(w_inter + w_slope*h_1, 0)/2.0
    w_2 = max(w_inter + w_slope*h_2, 0)/2.0
    w_3 = -max(w_inter + w_slope*h_3, 0)/2.0
    w_4 = max(w_inter + w_slope*h_4, 0)/2.0
    w_5 = -max(w_inter + w_slope*h_5, 0)/2.0
    w_6 = max(w_inter + w_slope*h_6, 0)/2.0


    set_points_element = np.array([
        # [0.0, 0.0, h_0],
        [l_1, w_1, h_1],
        [l_2, w_2, h_2],
        [l_3, w_3, h_3],
        [l_4, w_4, h_4],
        [l_5, w_5, h_5],
        [l_6, w_6, h_6]
    ])

    # set_points_element = np.array([
    #     [0.0        , 0.0 , height],
    #     [sign*5*l_10, -w_2, height],
    #     [sign*3*l_10,  w_2, height  + height_step*0.2],
    #     [sign*1*l_10, -w_2, height  + height_step*0.4],
    #     [-sign*1*l_10,  w_2, height + height_step*0.6],
    #     [-sign*3*l_10, -w_2, height + height_step*0.8],
    #     [-sign*5*l_10,  w_2, height + height_step*1.0],
    # ])

    # 5 distances: increas height with 0.2 of height step for every distance
    # This ensures a even ascend while traversing

    set_points = np.concatenate((set_points, set_points_element))

    height += height_step
    sign *= -1


# set_points = np.array([[0,0,min_height]])

# set_points_element = np.array([
#         [0, 0, 5.0],
#         [0, 0, 10.0]
#     ])

# set_points = np.concatenate((set_points, set_points_element))


# Add the final set point
final_set_point = np.array([
    [0.0, 0.0, max_height],
])
set_points = np.concatenate((set_points, final_set_point))

print(set_points)

# Running setting

frequency = 100 # Maximum around 250

time_step = 1.0 / frequency

set_point_counter = 0
step_counter = 0
transition_steps = 0
total_transition_time = 0
step_vector = np.empty(3)
def get_next_set_point(uav_time):
    global set_point_counter
    global step_counter
    global step_vector
    global transition_steps
    global total_transition_time

    # When finished, report the last setpoint
    if set_point_counter+1 >= len(set_points):
        # return set_points[len(set_points)-1]
        return None

    start_point = set_points[set_point_counter]
    end_point = set_points[set_point_counter+1]

    if step_counter == 0:
        vector = end_point - start_point
        distance = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        # rospy.loginfo("Distance to next setpoint: " + str(distance))
    
        transition_duration = distance / speed # 1.41421 s
        total_transition_time += transition_duration

        transition_steps = max(1, math.ceil(transition_duration / time_step))

        step_vector = vector / transition_steps
        # rospy.loginfo("step_vector: " + str(step_vector))
        
        next_set_point = start_point
        step_counter += 0.5
    elif step_counter == 1:
        next_set_point = start_point + step_vector*step_counter
        step_counter += 1
    elif step_counter == transition_steps-0.5:
        next_set_point = start_point + step_vector*step_counter
        step_counter += 0.5
    elif step_counter >= transition_steps:
        step_counter = 0
        set_point_counter += 1
        next_set_point = end_point
    else:
        next_set_point = start_point + step_vector*step_counter
        step_counter += 1

    

    return next_set_point

def run():   
    rospy.init_node('trajectory', anonymous=True)

    topic = 'visualization_marker_array'
    publisher = rospy.Publisher(topic, MarkerArray, queue_size=10)
    set_point_pub = rospy.Publisher("/set_point", Point, queue_size=10)
    signal_pub = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=1)

    rospy.loginfo("Starting trajectory module")

    markerArray = MarkerArray()

    # Landing platform cylinder
    helipad = Marker()
    helipad.header.frame_id = "/ocean"
    helipad.id = 0
    helipad.type = helipad.CYLINDER
    helipad.action = helipad.ADD
    helipad.scale.x = 1.0
    helipad.scale.y = 1.0
    helipad.scale.z = 0.1
    helipad.color.r = 0.04
    helipad.color.g = 0.8
    helipad.color.b = 0.04
    helipad.color.a = 0.3
    helipad.pose.orientation.w = 1.0
    helipad.pose.position.x = HELIPAD_POS_X
    helipad.pose.position.y = HELIPAD_POS_Y
    helipad.pose.position.z = HELIPAD_POS_Z

    # UAV sphere
    marker = Marker()
    marker.header.frame_id = "/ocean"
    marker.id = 1
    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0

    if len(markerArray.markers) == 0:
        markerArray.markers.append(helipad)
    else:
        markerArray.markers[0] = helipad

    uav_time = 0.0

    set_point_msg = Point()

    rate = rospy.Rate(frequency) # Hz
    while not rospy.is_shutdown():


        uav_set_point = get_next_set_point(uav_time)
        rospy.loginfo("Set_point: " + str(uav_set_point))

    
        if uav_set_point is not None:
            uav_time += time_step
            rospy.loginfo("Total transition time: " + str(total_transition_time))
            
            marker.pose.position.x = HELIPAD_POS_X + uav_set_point[0]
            marker.pose.position.y = HELIPAD_POS_Y + uav_set_point[1]
            marker.pose.position.z = HELIPAD_POS_Z + uav_set_point[2]

            if len(markerArray.markers) <= 1:
                markerArray.markers.append(marker)
            else:
                markerArray.markers[1] = marker
        
            # Publish the MarkerArray
            publisher.publish(markerArray)

            # Publish the setpoint
            set_point_msg.x = uav_set_point[0]
            set_point_msg.y = uav_set_point[1]
            set_point_msg.z = uav_set_point[2]
            set_point_pub.publish(set_point_msg)

        else:
            # Indicate that the trajectory is finished
            signal_pub.publish(Empty())
            break

        # if uav_time >= 3.0:
        #     break

        rate.sleep()
    
    

if __name__ == '__main__':
    run()