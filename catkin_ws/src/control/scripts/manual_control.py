#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Empty, Float32
from geometry_msgs.msg import Twist


prev_teleop_msg = Twist()

def teleop_callback(data):
    global prev_teleop_msg
    prev_teleop_msg = data


def main():
    rospy.init_node('manual_control', anonymous=True)


    take_off = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    land = rospy.Publisher("/ardrone/land", Empty, queue_size=10)
    controller = rospy.Publisher("/cmd_vel", Twist, queue_size=1000)

    rospy.Subscriber("key_vel", Twist, teleop_callback)


    empty_msg = Empty()
    control_msg = Twist()

    zero_msg_count = 0

    rate = rospy.Rate(100) # Hz
    while not rospy.is_shutdown():


        if prev_teleop_msg == Twist():
            zero_msg_count += 1
        else:
            zero_msg_count = 0
        
        if zero_msg_count <= 1:
            # Only send control message if something new on the input


            control_msg.linear.x = prev_teleop_msg.linear.x
            control_msg.linear.y = prev_teleop_msg.linear.y
            control_msg.linear.z = prev_teleop_msg.linear.z

            control_msg.angular.x = 0
            control_msg.angular.y = 0
            control_msg.angular.z = prev_teleop_msg.angular.z


            if (prev_teleop_msg.angular.x):
                take_off.publish(empty_msg)
            elif (prev_teleop_msg.angular.y):
                land.publish(empty_msg)
            else:
                controller.publish(control_msg)


        rate.sleep()

if __name__ == '__main__':
    main()