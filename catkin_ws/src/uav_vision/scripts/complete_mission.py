#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Bool


from timer import Timer

# Quadcopter States
STATE_INIT = 0
STATE_TAKEOFF = 1
STATE_HOVER = 2
STATE_LANDING = 3
STATE_PHOTOTWIRL = 4
STATE_MOVING = 5
STATE_ERROR = 6
STATE_IDLE = 7

empty = Empty()
desired_pose = Twist()
current_pose = Twist()
timer = Timer()
timeout_timer = Timer()
state = STATE_INIT
init_complete = False
landing_complete = False
phototwirl_complete = False
photos_taken = 0



#############
# Callbacks #
#############
def estimate_callback(data):
    global current_pose
    global init_complete
    print('received estimate')
    init_complete = True
    current_pose = data

def landing_complete_callback(data):
    global landing_complete
    print('received landing complete')
    landing_complete = True


def main():
    rospy.init_node('complete_mission',anonymous=True)

    global init_complete
    global landing_complete
    global phototwirl_complete
    global photos_taken
    global empty
    global state
    pub_desired_pose = rospy.Publisher("/set_point", Twist, queue_size=1)
    pub_start_takeoff = rospy.Publisher("/ardrone/takeoff", Empty, queue_size=10)
    pub_start_automated_landing = rospy.Publisher('/initiate_automated_landing', Empty, queue_size=1)
    pub_save_front_camera_photo = rospy.Publisher('/take_still_photo_front',Empty, queue_size=1)

    rospy.Subscriber('/estimate/dead_reckoning', Twist, estimate_callback)
    rospy.Subscriber('/ardrone/land', Empty, landing_complete_callback)


    timer.start(10)
    while not rospy.is_shutdown():
        if state == STATE_INIT:
            if init_complete and timer.is_timeout():
                timer.stop()
                state = STATE_TAKEOFF
                pub_start_takeoff.publish(empty)

        if state == STATE_TAKEOFF:


if __name__ == "__main__":
    main()
