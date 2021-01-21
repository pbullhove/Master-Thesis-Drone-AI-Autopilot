#!/usr/bin/env python

import rospy

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import numpy as np
from scipy.misc import imsave
import os
from scipy.spatial.transform import Rotation as R


print(os.getcwd())
os.chdir('/home/thomas/master_project/investigations')
print(os.getcwd())


previous_image = None
prev_rel_position = [None, None, None, None]

PLATFORM_OFFSET_X = 2
PLATFORM_OFFSET_Y = 2
PLATFORM_OFFSET_Z = 0


def position_callback(data):
    global prev_rel_position
    # Transform ground truth in body frame wrt. world frame to body frame wrt. landing platform

    ##########
    # 0 -> 2 #
    ##########

    # Position
    p_x = data.pose.pose.position.x
    p_y = data.pose.pose.position.y
    p_z = data.pose.pose.position.z

    # Translation of the world frame to body frame wrt. the world frame
    d_0_2 = np.array([p_x, p_y, p_z])

    # Orientation
    q_x = data.pose.pose.orientation.x
    q_y = data.pose.pose.orientation.y
    q_z = data.pose.pose.orientation.z
    q_w = data.pose.pose.orientation.w

    # Rotation of the body frame wrt. the world frame
    r_0_2 = R.from_quat([q_x, q_y, q_z, q_w])
    r_2_0 = r_0_2.inv()
    

    ##########
    # 0 -> 1 #
    ##########
    
    # Translation of the world frame to landing frame wrt. the world frame
    offset_x = 1.0
    offset_y = 0.0
    offset_z = 0.54
    d_0_1 = np.array([offset_x, offset_y, offset_z])

    # Rotation of the world frame to landing frame wrt. the world frame
    # r_0_1 = np.identity(3) # No rotation, only translation
    r_0_1 = np.identity(3) # np.linalg.inv(r_0_1)


    ##########
    # 2 -> 1 #
    ##########
    # Transformation of the body frame to landing frame wrt. the body frame
    
    # Translation of the landing frame to bdy frame wrt. the landing frame
    d_1_2 = d_0_2 - d_0_1

    # Rotation of the body frame to landing frame wrt. the body frame
    r_2_1 = r_2_0

    yaw = r_2_1.as_euler('xyz')[2] # In radians
    r_2_1_yaw = R.from_euler('z', yaw)

    # Translation of the body frame to landing frame wrt. the body frame
    # Only yaw rotation is considered
    d_2_1 = -r_2_1_yaw.apply(d_1_2)

    # Translation of the landing frame to body frame wrt. the body frame
    # This is more intuitive for the controller
    prev_rel_position = np.concatenate((-d_2_1, [yaw]))


def video_callback(data):
    global previous_image
    previous_image = data


def save_image():
    if previous_image == None:
        return False

    this_image = previous_image

    rospy.loginfo("Saving image")

    height = this_image.height # height = 360
    width = this_image.width # width = 640
    depth = 3
    num_elements = height*width*3

    image_array = np.zeros((num_elements))

    for i in range(num_elements):
        image_array[i] = float(ord(this_image.data[i]))
    image_array.resize(height,width,depth)


    filename = 'image_2_corners_long_side'
    filetype = '.jpg' # or 'png'
    filepath = 'dataset/' + filename + filetype
    imsave(filepath, image_array)

    f = open("position_2_corners_long_side.txt", "w")
    x_pos = str(prev_rel_position[0])
    y_pos = str(prev_rel_position[1])
    z_pos = str(prev_rel_position[2])
    yaw = str(prev_rel_position[3])
    f.write('x_pos: ' + x_pos + ' | y_pos: ' + y_pos + ' | z_pos: ' + z_pos + ' | yaw: ' + yaw)
    f.close()

    return True


def run():   
    
    rospy.init_node('save_image', anonymous=True)

    rospy.Subscriber('ardrone/bottom/image_raw', Image, video_callback)
    rospy.Subscriber('ground_truth/state', Odometry, position_callback)

    rospy.loginfo("Starting saving of images")

    rate = rospy.Rate(0.5) # Hz
    while not rospy.is_shutdown():
        # Do work
        if save_image():
            break
        
        rate.sleep()
    
    

if __name__ == '__main__':
    run()
