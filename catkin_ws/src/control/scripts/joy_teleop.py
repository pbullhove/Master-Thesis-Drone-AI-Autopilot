#! /usr/bin/env python

"""
Module for manual control using PS4-controller for quadcopter control.
"""

import rospy
from std_msgs.msg import String, Empty, Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

import numpy as np

# Higher sensitivity, higher speed
sensitivity_x_y = 2.0
sensitivity_z = 0.5
sensitivity_yaw = 1.0

set_point_linear_x = 0.0
set_point_linear_y = 0.0
set_point_linear_z = 2.0
set_point_angular_z = 0.0

set_points = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

# Start in manual control
manual_control = True
set_point_control = False

def teleop_callback(data):
    global set_points
    global manual_control
    global set_point_control

    # axes [8] = [left_js left-right(0), left_js up-down(1), L2(2), right_js left-right(3), right_js up-down(4), R2(5), upper_js left-right(6), upper_js up-down(7)]
    # buttons [13] = [cross(0), circle(1), triangle(2), square(3), L1(4), R1(5), L2(6), R2(7), share(8), options(9), start(10), js_left(11), js_right(12)]
    axes = data.axes
    buttons = data.buttons

    left_js_horizontal = axes[0]
    left_js_vertical = axes[1]

    right_js_horizontal = axes[3]
    right_js_vertical = axes[4]

    if buttons[0]:
        # "Cross"-button -> Take off
        pub_take_off.publish(Empty())

    if buttons[1]:
        # "Circle"-button -> Land
        pub_land.publish(Empty())

    if buttons[2]:
        # "Triangle"-botton -> Initiate mission and start data collection
        pub_initiate_mission.publish(Empty())

    if buttons[3]:
        # "Square"-button -> Take still photo and stop data collection
        pub_take_still_photo.publish(Empty())

    if buttons[4]:
        # "L1"-button -> Toggle between manual control and set point control
        if manual_control:
            manual_control = False
            set_point_control = True
            rospy.loginfo("Set point controll on")
            pub_pid_on_off.publish(Bool(True))

        elif set_point_control:
            set_point_control = False
            manual_control = True
            rospy.loginfo("Manual control on")
            set_points = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # Reset set point to hover at 1m
            pub_pid_on_off.publish(Bool(False))

    if buttons[5]:
        # "R1"-button -> Emergency landing
        pub_emergency.publish(Empty())

    if buttons[10]:
        # "Start"-button -> Initiate automated landing
        rospy.loginfo("Automated landing command sent")

        pub_initiate_automated_landing.publish(Empty())

    if manual_control:
        # Publish velocity command directly to the quadcopter
        control_msg = Twist()
        control_msg.linear.x = right_js_vertical*sensitivity_x_y
        control_msg.linear.y = right_js_horizontal*sensitivity_x_y
        control_msg.linear.z = left_js_vertical*sensitivity_z
        control_msg.angular.z = left_js_horizontal*sensitivity_yaw
        pub_controller.publish(control_msg)


    if set_point_control:
        # Publish set point to the PID controller

        amplifier = 0.01
        yaw_amplifier = 1
        # yaw_amplifier = 0

        set_points += np.array([amplifier*right_js_vertical*sensitivity_x_y,  amplifier*right_js_horizontal*sensitivity_x_y,   amplifier*left_js_vertical*sensitivity_z,
                                0.0,                                0.0,                                    yaw_amplifier*left_js_horizontal*sensitivity_yaw])

        if set_points[5] > 180:
            set_points[5] = -180
        elif set_points[5] <= -180:
            set_points[5] = 180


        set_point_msg = Twist()
        set_point_msg.linear.x = set_points[0]
        set_point_msg.linear.y = set_points[1]
        set_point_msg.linear.z = set_points[2]
        set_point_msg.angular.z = set_points[5]
        pub_set_point.publish(set_point_msg)


def main():
    global pub_take_off
    global pub_land
    global pub_emergency
    global pub_controller
    global pub_take_still_photo
    global pub_initiate_mission
    global pub_initiate_automated_landing

    global pub_set_point
    global pub_pid_on_off

    pub_take_off = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_land = rospy.Publisher("/ardrone/land", Empty, queue_size=10)
    pub_emergency = rospy.Publisher("/ardrone/reset", Empty, queue_size=10)
    pub_controller = rospy.Publisher("/cmd_vel", Twist, queue_size=1000)

    pub_set_point = rospy.Publisher("/set_point", Twist, queue_size=10)

    pub_take_still_photo = rospy.Publisher("/take_still_photo", Empty, queue_size=10)
    pub_initiate_mission = rospy.Publisher("/initiate_mission", Empty, queue_size=10)
    pub_initiate_automated_landing = rospy.Publisher("/initiate_automated_landing", Empty, queue_size=10)

    pub_pid_on_off = rospy.Publisher("/pid_on_off", Bool, queue_size=10)

    rospy.Subscriber("joy", Joy, teleop_callback)


    rospy.init_node('joy_teleop', anonymous=True)
    rospy.loginfo("Joystick teleoperation ready")


    rospy.loginfo("Manual control on")
    pub_pid_on_off.publish(Bool(False))

    rospy.spin()


if __name__ == '__main__':
    main()
