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
init_complete = False
landing_complete = False
phototwirl_complete = False
photos_taken = 0



#############
# Callbacks #
#############
def estimate_callback(data):
    global current_pose
    current_pose = data

def landing_complete_callback(data):
    global landing_complete
    landing_complete = True

def init_complete_callback(data):
    global init_complete
    init_complete = True


def main():
    rospy.init_node('complete_mission',anonymous=True)


    #pub_init_drone = rospy.Publisher('/')
    pub_desired_pose = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_start_takeoff = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_start_automated_landing = rospy.Publisher('/initiate_automated_landing', Empty, queue_size=1)
    pub_save_front_camera_photo = rospy.Publisher('/take_still_photo_front',Empty, queue_size=1)

    rospy.Subscriber('/estimate/dead_reckoning', Twist, estimate_callback)
    rospy.Subscriber('/ardrone/land', Empty, landing_complete_callback)



    while not rospy.is_shutdown():
        pass



if __name__ == "__main__":
    main()
