#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool


import FSM_states
from timer import Timer


desired_pose = Twist()
current_pose = Twist()
timer = Timer()
timeout_timer = Timer()
current_state = STATE_INIT



#############
# Callbacks #
#############
def estimate_callback(data):
    global current_pose
    current_pose = data



def main():
    rospy.init_node('complete_mission',anonymous=True)


    while not rospy.is_shutdown():
        if not timer.is_timeout():
            pass
        else:
            print('timer timeout')
            break



if __name__ == "__main__":
    main()
