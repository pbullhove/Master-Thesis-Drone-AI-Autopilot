#!/usr/bin/env python
# coding=utf-8

# from __future__ import absolute_import, division, print_function, unicode_literals

# standard packages
import numpy as np
import random
import copy
import math
from tqdm import tqdm
import time
import json
import os
import sys
from collections import deque
import tensorflow as tf
from scipy.spatial.transform import Rotation as R
from keras import backend as K


# ros oriented packages 
import roslib
import rospy
from std_srvs.srv import Empty as empty
from std_msgs.msg import Empty, Bool, Int32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

# self-implemented modules 
from DDPGAgent import DDPGAgent

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress deprecation warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # forces tensorflow to use CPU
sys.dont_write_bytecode = True

## Initialize node, publishers and services
rospy.init_node('control_script')
rate_hz = 3
rate = rospy.Rate(rate_hz)

pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1000)
pub_land = rospy.Publisher('/ardrone/land', Empty, queue_size=1000)
pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)

reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', empty)


# Define the system: behavior and helping functions
class DroneState():
    # Initializes internal variables
    def __init__(self):
        # get size of state and action
        self.state_size_hover = 9 # x y z vx vy vz ax ay az 
        self.action_size_hover = 3 # velocity in x y z
        self.state_size_descend = 10 # x y z vx vy vz ax ay az 
        self.action_size_descend = 4 # velocity in x y z

        # Data received from Odometry
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        # Data received from Navdata
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0

        # Data received from Imu
        self.wx = 0.0
        self.wy = 0.0
        self.wz = 0.0

        # previous action are also stored in state array
        self.a_prev = [0.0] * self.action_size_hover

        # state sub arrays
        self.drone_state_odometry = [self.x, self.y, self.z] # 3
        self.drone_yaw_estimate = 0.0
        self.drone_state_navdata = [self.vx, self.vy, self.vz, self.ax, self.ay, self.az, self.rot_x, self.rot_y, self.rot_z] # 9
        self.drone_state_imu = [self.wx, self.wy, self.wz] # 3
        self.drone_state_previous_action = self.a_prev # self.action_size
        
        # complete state array
        self.drone_state = self.drone_state_odometry + self.drone_state_navdata + self.drone_state_imu + self.drone_state_previous_action

        # drone estimate status. 0: no estimate, 1: ellipse estimate (drone far away from platform), 
        # 2: helipad estimate (close), 3: imu estimate (using estimate from imu)
        self.curr_drone_estimate_status = 0 # initialize to 0 or -1?


        ##################### HOVER
        # Position controller goal
        self.x_goal_hover = 0.0
        self.y_goal_hover = 0.0
        self.z_goal_hover = 2.0

        # saturation of actions
        self.action_low_hover = -1.0
        self.action_high_hover = 1.0

        # bounds of various states. showcasing the state space the drone can find itself in in the environment
        self.minx_hover = self.x_goal_hover - 2.0
        self.maxx_hover = self.x_goal_hover + 2.0
        self.miny_hover = self.y_goal_hover - 2.0
        self.maxy_hover = self.y_goal_hover + 2.0
        self.minz_hover = self.z_goal_hover - 1.0
        self.maxz_hover = self.z_goal_hover + 1.0

        ##################### DESCEND
        # Position controller goal
        self.x_goal_descend = 0.0
        self.y_goal_descend = 0.0
        self.z_goal_descend = 0.4
        self.rotz_goal_descend = 0.0

        # saturation of actionss
        self.action_low_descend = -0.4
        self.action_high_descend = 0.4

        # bounds of various states. showcasing the state space the drone can find itself in in the environment
        self.minx_descend = self.x_goal_descend - 0.5
        self.maxx_descend = self.x_goal_descend + 0.5
        self.miny_descend = self.y_goal_descend - 0.5
        self.maxy_descend = self.y_goal_descend + 0.5
        self.minz_descend = self.z_goal_descend - 2.0
        self.maxz_descend = self.z_goal_descend + 2.0
        self.minrotz_descend = self.rotz_goal_descend - 20
        self.maxrotz_descend = self.rotz_goal_descend + 20

        # state of the fsm. 1: hovering, 2: descending, 3: landing, 4: landed
        self.fsm_state = 1

        # create agent
        self.agent = DDPGAgent(self.state_size_hover, self.action_size_hover, self.action_low_hover, self.action_high_hover)

    def print_state(self, state):
        print("x: {}, \t y: {}, \t z: {}".format(state[0], state[1], state[2]))
        # print("vx: {}, \t vy: {}, \t vz: {}".format(state[3], state[4], state[5]))
        # print("ax: {}, \t ay: {}, \t az: {}".format(state[6], state[7], state[8]))
        # print("rotx: {}, \t roty: {}, \t rotz: {}".format(state[8], state[10], state[11]))
        # print("wx: {}, \t wy: {}, \t wz: {}".format(state[12], state[13], state[14]))
        # print("a_prev_x: {}, \t a_prev_y: {}, \t a_prev_z: {}".format(state[15], state[16], state[17]))

    # Reduces the original state vector. Only preserves states that we actually use
    def reduce_state_array_hover(self, state):
        #            x       y           z       vx          vy      vz        ax         ay        az   
        return [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]]
    
    # Reduces the original state vector. Only preserves states that we actually use
    def reduce_state_array_descend(self, state):
        #            x       y           z       vx          vy      vz        ax         ay        az       rotz
        return [state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8], state[11]]

    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    # normalizes a (state) s, that originally has bounds [mins, maxs] and returns s with bounds [a,b]
    def normalize_state(self, s, mins, maxs, a, b):
        return (b - a) * (s - mins) / (maxs - mins) + a


    # normalize state array to range of [-1,1]
    def normalize_state_array_hover(self, state_reduced):
        norm_list = []
        # normalize x y z according to boundaries and setpoint in the environment. If setpoint is (0,0,2) then 
        # x goes from -2 to 2, same with y, while z goes from -1 to 1.
        norm_x = self.normalize_state(state_reduced[0], self.minx_hover - abs(self.x_goal_hover), self.maxx_hover - abs(self.x_goal_hover), -1, 1) 
        norm_y = self.normalize_state(state_reduced[1], self.miny_hover - abs(self.y_goal_hover), self.maxy_hover - abs(self.y_goal_hover), -1, 1)
        norm_z = self.normalize_state(state_reduced[2], self.minz_hover - abs(self.z_goal_hover), self.maxz_hover - abs(self.z_goal_hover), -1, 1)
        norm_list.append(norm_x)
        norm_list.append(norm_y)
        norm_list.append(norm_z)

        # min and max for velocities come from testing. See log 24.2 and maxminlist from 11.3
        norm_vx = self.normalize_state(state_reduced[3], -1500, 1500, -1, 1) # -1439, 1402
        norm_vy = self.normalize_state(state_reduced[4], -1550, 1450, -1, 1) # -1511, 1422
        norm_vz = self.normalize_state(state_reduced[5], -550, 550, -1, 1) # -505, 524
        norm_list.append(norm_vx)
        norm_list.append(norm_vy)
        norm_list.append(norm_vz)

        # min and max for accelerations come from testing. See log 24.2 and maxminlist from 11.3
        norm_ax = self.normalize_state(state_reduced[6], -0.3, 0.3, -1, 1) 
        norm_ay = self.normalize_state(state_reduced[7], -0.3, 0.3, -1, 1) 
        norm_az = self.normalize_state(state_reduced[8], 0.5, 1.5, -1, 1) 
        norm_list.append(norm_ax)
        norm_list.append(norm_ay)
        norm_list.append(norm_az)

        return norm_list

    # normalize state array to range of [-1,1]
    def normalize_state_array_descend(self, state_reduced):
        norm_list = []
        # normalize x y z according to boundaries and setpoint in the environment. If setpoint is (0,0,2) then 
        # x goes from -2 to 2, same with y, while z goes from -1 to 1.
        norm_x = self.normalize_state(state_reduced[0], self.minx_descend - abs(self.x_goal_descend), self.maxx_descend - abs(self.x_goal_descend), -1, 1) 
        norm_y = self.normalize_state(state_reduced[1], self.miny_descend - abs(self.y_goal_descend), self.maxy_descend - abs(self.y_goal_descend), -1, 1)
        norm_z = self.normalize_state(state_reduced[2], self.minz_descend - abs(self.z_goal_descend), self.maxz_descend - abs(self.z_goal_descend), -1, 1)
        norm_list.append(norm_x)
        norm_list.append(norm_y)
        norm_list.append(norm_z)

        # min and max for velocities come from testing. See log 24.2 and maxminlist from 11.3
        norm_vx = self.normalize_state(state_reduced[3], -1500, 1500, -1, 1) # -1439, 1402
        norm_vy = self.normalize_state(state_reduced[4], -1550, 1450, -1, 1) # -1511, 1422
        norm_vz = self.normalize_state(state_reduced[5], -550, 550, -1, 1) # -505, 524
        norm_list.append(norm_vx)
        norm_list.append(norm_vy)
        norm_list.append(norm_vz)

        # min and max for accelerations come from testing. See log 24.2 and maxminlist from 11.3
        norm_ax = self.normalize_state(state_reduced[6], -0.3, 0.3, -1, 1) 
        norm_ay = self.normalize_state(state_reduced[7], -0.3, 0.3, -1, 1) 
        norm_az = self.normalize_state(state_reduced[8], 0.5, 1.5, -1, 1) 
        norm_list.append(norm_ax)
        norm_list.append(norm_ay)
        norm_list.append(norm_az)

        # min and max for rotz from observing that it is never outside minrotz or maxrotz (45 deg)
        norm_rotz = self.normalize_state(state_reduced[9], self.minrotz_descend - abs(self.rotz_goal_descend), self.maxrotz_descend - abs(self.rotz_goal_descend), -1, 1)  
        norm_list.append(norm_rotz)        

        return norm_list

    # Rotates a 3D vector (xyz, vxvyvz, axayaz whatever) yaw_angle degrees about z
    def rotate_vector(self, vector, yaw_angle):
        r = R.from_euler('z', yaw_angle, degrees=True) 
        rotated_vector = r.apply(vector)
        return rotated_vector

    # Sets x, y and z
    def set_drone_state_odometry(self, state_odometry):
        self.drone_state_odometry = state_odometry

    def set_drone_state_yaw_estimate(self, yaw_estimate):
        self.drone_yaw_estimate = yaw_estimate
    
    # Sets vx, vy, vz, ax, ay, az, rotx, roty, rotz
    def set_drone_state_navdata(self, state_navdata):
        self.drone_state_navdata = state_navdata

    # Sets wx, wy, wz
    def set_drone_state_imu(self, state_imu):
        self.drone_state_imu = state_imu

    # Sets previos action
    def set_drone_state_previous_action(self, a_prev):
        self.drone_state_previous_action = a_prev

    # Sets x, y, z, vx, vy, vz, ax, ay, az, rotx, roty, rotz, wx, wy, wz, a_prev
    def set_drone_state(self, state_odometry, state_navdata, state_imu, a_prev):
        self.set_drone_state_odometry(state_odometry)
        self.set_drone_state_navdata(state_navdata)
        self.set_drone_state_imu(state_imu)
        self.set_drone_state_previous_action(a_prev)

    # Sets drone estimate status
    def set_drone_estimate_status(self, status):
        self.curr_drone_estimate_status = status

    # Gets x, y and z
    def get_drone_state_odometry(self):
        return self.drone_state_odometry

    # Gets vx, vy, vz, ax, ay, az, rotx, roty, rotz
    def get_drone_state_navdata(self):
        return self.drone_state_navdata

    # Gets wx, wy, wz
    def get_drone_state_imu(self):
        return self.drone_state_imu

    # Gets a_prev
    def get_drone_state_previous_action(self):
        return self.drone_state_previous_action

    # Gets x, y, z, vx, vy, vz, ax, ay, az, rotx, roty, rotz, wx, wy, wz, a_prev
    def get_drone_state(self):
        state_odometry = self.get_drone_state_odometry()
        state_yaw_estimate = self.drone_yaw_estimate
        state_navdata = self.get_drone_state_navdata()
        state_navdata[-1] = state_yaw_estimate

        state_imu = self.get_drone_state_imu()
        a_prev = self.get_drone_state_previous_action()
        return state_odometry + state_navdata + state_imu + a_prev

    # Sets drone estimate status
    def get_drone_estimate_status(self):
        return self.curr_drone_estimate_status

    # Checks if the drone is not in a valid area
    def out_of_bounds(self, state):
        is_out_of_bounds = False
        if state[0] < self.minx_hover or state[0] > self.maxx_hover:
            print("out of bounds in x")
            is_out_of_bounds = True
        if state[1] < self.miny_hover or state[1] > self.maxy_hover:
            print("out of bounds in y")
            is_out_of_bounds = True
        if state[2] < self.minz_hover or state[2] > self.maxz_hover:
            print("out of bounds in z")
            is_out_of_bounds = True

        return is_out_of_bounds

    # Resets the world
    def reset(self):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_simulation()
        except(rospy.ServiceException) as e:
            print("reset_world failed!", e)
        print("\n called reset()")

        self.set_drone_state([0.0]*3, [0.0]*9, [0.0]*3, [0.0]*self.action_size_hover)
        return self.get_drone_state()

    # Transports the drone to a random start state within the valid area
    def set_start_state(self):
        # start_position = [random.uniform(env.minx, env.maxx), random.uniform(env.miny, env.maxy), random.uniform(env.minz, env.maxz)]
        # if start_position[2] <= 0.1:
        #     start_position[2] = 0.2
        # start_position = [-2,-2,3]
        start_position = [1,0,4]
        print("start state is: ({},{},{})".format(start_position[0], start_position[1], start_position[2]))
        
        state_msg = ModelState()
        state_msg.model_name = 'quadrotor'
        state_msg.pose.position.x = start_position[0]
        state_msg.pose.position.y = start_position[1]
        state_msg.pose.position.z = start_position[2]
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state(state_msg)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

        self.set_drone_state(start_position, [0.0]*9, [0.0]*3, [0.0]*self.action_size_hover)
        return self.get_drone_state()

    # Applies an action to the drone and returns the next state, the reward and if the episode is done
    def step_hover(self, state_reduced, action):

        # unfreeze simulator so it can apply the action
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            unpause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"

        # get action from actor
        u_x, u_y, u_z  = action[0], action[1], action[2]

        command = Twist()
        command.linear.x = u_x
        command.linear.y = u_y
        command.linear.z = u_z
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        pub_action.publish(command)

        rate.sleep()

        # get the new state we have arrived at AFTER action is done, and get the reward
        next_state = self.get_drone_state()
        
        # freeze simulator so the simulator state does not change during calculations 
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            pause()
        except (rospy.ServiceException) as e:
            print "rospause failed!"

        reward, done, pos_error = self.get_reward_hover(action, next_state)

        return next_state, reward, done, pos_error
    
    # Applies an action to the drone and returns the next state, the reward and if the episode is done
    def step_descend(self, state_reduced, action):

        # unfreeze simulator so it can apply the action
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            unpause()
        except (rospy.ServiceException) as e:
            print "/gazebo/pause_physics service call failed"

        # get action from actor
        u_x, u_y, u_z, u_rotz = action[0], action[1], action[2], action[3]

        command = Twist()
        command.linear.x = u_x
        command.linear.y = u_y
        command.linear.z = u_z
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = u_rotz
        pub_action.publish(command)

        rate.sleep()

        # get the new state we have arrived at AFTER action is done, and get the reward
        next_state = self.get_drone_state()
        print("next_state")
        env.print_state(next_state)
        
        # freeze simulator so the simulator state does not change during calculations 
        # rospy.wait_for_service('/gazebo/pause_physics')
        # try:
        #     pause()
        # except (rospy.ServiceException) as e:
        #     print "rospause failed!"

        reward, done, pos_error, pos_error_aug = self.get_reward_descend(action, next_state)

        return next_state, reward, done, pos_error, pos_error_aug

    # reward function that penalizes the drone for not hovering at z=2
    def get_reward_hover(self, current_action, next_state):
        reward = 0.0

        # flag indicating that the episode is done
        done = False

        delta_x = next_state[0] - self.x_goal_hover
        delta_y = next_state[1] - self.y_goal_hover
        delta_z = next_state[2] - self.z_goal_hover

        sigma = np.sqrt(0.05/2)
        pos_error = (delta_x**2 + delta_y**2 + delta_z**2)

        reward = np.exp( - pos_error / (2 * sigma **2) )

        # if self.out_of_bounds(next_state) and train_agent:
        #     done = True

        return reward, done, pos_error

    # reward function that penalizes the drone for not hovering at z=2
    def get_reward_descend(self, current_action, next_state):
        reward = 0.0

        # flag indicating that the episode is done
        done = False

        delta_x = next_state[0] - self.x_goal_descend
        delta_y = next_state[1] - self.y_goal_descend
        delta_z = next_state[2] - self.z_goal_descend
        delta_rotz = next_state[11] - self.rotz_goal_descend
        print("delta_x", delta_x)
        print("delta_y", delta_y)
        print("delta_z", delta_z)
        print("delta_rotz", delta_rotz)
        # delta_x = self.normalize_state(delta_x, self.minx, self.maxx, -1, 1) 
        # delta_y = self.normalize_state(delta_y, self.miny, self.maxy, -1, 1)
        # delta_z = self.normalize_state(delta_z, self.minz, self.maxz, -1, 1)
        # delta_rotz = self.normalize_state(delta_rotz, self.minrotz, self.maxrotz, -1, 1)  
        delta_rotz = delta_rotz / 20

        # delta_rotz = delta_rotz / 90 # scaling rotz delta down so pos_error does not blow up

        sigma = np.sqrt(0.05/2)
        pos_error = (delta_x**2 + delta_y**2 + delta_z**2)
        pos_error_aug = (delta_x**2 + delta_y**2 + delta_z**2 + delta_rotz**2)

        reward = np.exp( - pos_error_aug / (2 * sigma **2) )

        # if pos_error < 0.2:
        #     reward = np.exp( - pos_error / (2 * sigma **2) ) - \
        #         factor_action * np.linalg.norm(current_action) 
        # else:
        #     reward = np.exp( - pos_error / (2 * sigma **2) )            

        if self.out_of_bounds(next_state) and train_agent:
            done = True

        return reward, done, pos_error, pos_error_aug


# Define the environment and controller
env = DroneState() 

## Helping functions 

# For drone takeoff
def takeoff():
    while pub_takeoff.get_num_connections() < 1:
        rospy.loginfo_throttle(2, "Waiting for subscribers on /ardrone/takeoff ..")
        rospy.sleep(0.1)
    pub_takeoff.publish(Empty())

# For drone landing
def land():
    # while pub_takeoff.get_num_connections() < 1:
    #     rospy.loginfo_throttle(2, "Waiting for subscribers on /ardrone/takeoff ..")
    #     rospy.sleep(0.1)
    pub_land.publish(Empty())

# Callback: setting x y z
# http://docs.ros.org/melodic/api/nav_msgs/html/msg/Odometry.html
def cb_drone_state_odometry(data):
    env.set_drone_state_odometry([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])

def cb_drone_estimate_thomas(data):
    env.set_drone_state_odometry([data.linear.x, data.linear.y, data.linear.z])
    env.set_drone_state_yaw_estimate(data.angular.z)

def cb_drone_estimate_status_thomas(data):
    env.set_drone_estimate_status = data

# Callback: setting vx vy vz ax ay az rotx roty rotz
# http://docs.ros.org/indigo/api/ardrone_autonomy/html/msg/Navdata.html
def cb_drone_state_navdata(data):
    env.set_drone_state_navdata([data.vx, data.vy, data.vz, data.ax, data.ay, data.az, data.rotX, data.rotY, data.rotZ])

# Callback: setting wx wy wz
# http://docs.ros.org/api/sensor_msgs/html/msg/Imu.html
def cb_drone_state_imu(data):
    env.set_drone_state_imu([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])

# Makes the subscribers
def init_subscribers(use_platform_estimate):
    if use_platform_estimate:
        # rospy.Subscriber("/drone_estimate", Point, cb_drone_estimate_thomas)
        rospy.Subscriber("/estimate/dead_reckoning", Twist, cb_drone_estimate_thomas)
        rospy.Subscriber("/curr_drone_estimate_status", Int32, cb_drone_estimate_status_thomas)
    else:
        rospy.Subscriber("/ground_truth/state", Odometry, cb_drone_state_odometry)
    rospy.Subscriber("/ardrone/navdata", Navdata, cb_drone_state_navdata)
    rospy.Subscriber("/ardrone/imu", Imu, cb_drone_state_imu)

# deciding which channel we subscribe to for getting x y z position of the drone. remember to run the position_estimate.py code if merge=True
use_platform_estimate = True

# deciding if we train the agent or if we test it
train_agent = False

if train_agent == False:
    # actor_imported_hover = tf.keras.models.load_model('/home/danie/catkin_ws/src/ddpg/src/results/may5_2/actor_model.hdf5')
    # actor_imported_descend = tf.keras.models.load_model('/home/danie/catkin_ws/src/ddpg/src/results/may12_2/actor_model.hdf5')

    actor_imported_hover = tf.keras.models.load_model('/home/elias/master_project/catkin_ws/src/ddpg_controller/scripts/results/may5_2/actor_model.hdf5')
    actor_imported_descend = tf.keras.models.load_model('/home/elias/master_project/catkin_ws/src/ddpg_controller/scripts/results/may12_2/actor_model.hdf5')

max_episodes = 10000 # number of episodes to train drone in
max_total_steps = 50e3 # max number of steps in total for the whole training period
max_steps = 1000 # max number of steps in each training episode

init_subscribers(use_platform_estimate)
reward_array = [] # array storing rewards in each episode
number_of_steps_array = [] # array storing number of steps in each episode
q_value_array = [] # array storing q values from critic
critic_loss_array = [] # array storing loss from critic
state_recorder = [] # array storing xyz for plotting results


def run_episode(max_steps, training):
    total_reward = 0
    done = False
    steps_completed = 0 
    if not use_platform_estimate: # if using estimate we use manual control to start drone
        state = env.reset()
        takeoff()
        state = env.set_start_state()

    time_start_of_episode = rospy.get_rostime()
    now = rospy.get_rostime()
    print("start episode")
    state = env.get_drone_state()
    prev_pos_error = 100 # storing the reward in the previous timestep
    hovered_in_place_counter = 0

    for step_number in range(max_steps):
        now = rospy.get_rostime()

        print("state")
        env.print_state(state)

        if env.fsm_state == 1:
            print "01 prev_pos_error:" + str(prev_pos_error)

            yaw = state[11] # extract yaw angle

            # build state vector. reduce state space to x y z vx vy vz ax ay az
            state_reduced = env.reduce_state_array_hover(state)

            # subtract goal states from xyz to get relative position to setpoint
            state_reduced[0] = state_reduced[0] - env.x_goal_hover
            state_reduced[1] = state_reduced[1] - env.y_goal_hover
            state_reduced[2] = state_reduced[2] - env.z_goal_hover
            
            # rotate pos, vel and acc vectors -yaw degrees about z in state array so they are expressed in body coordinates
            # state_reduced[0:3] = env.rotate_vector(state_reduced[0:3], -yaw)

            # normalize each vector inside the state array
            state_reduced = env.normalize_state_array_hover(state_reduced) 

            # reformat for neat storage
            state_reduced = np.reshape(state_reduced, [1, env.state_size_hover]) 

            # choose action
            pure_action = actor_imported_hover.predict(state_reduced)[0]
            action = np.clip(pure_action, -0.4, 0.4)
            # action = np.clip(pure_action, env.agent.action_low, env.agent.action_high)
            # if close to hover, downscale action
            print("02 prev_pos_error", prev_pos_error)
            if prev_pos_error < 0.03:
                print("DOWNSCALE BENCHOD")
                action = 0.25 * action
                hovered_in_place_counter += 1
                if hovered_in_place_counter > 12: # mer enn 12 steps vil si 4 sekunder (3Hz)
                    env.fsm_state = 2
                    print("CHANGED GOAL")
                    hovered_in_place_counter = 0

            # if env.z_goal <= 0.0:
            #     land()
            #     print("landinggggggggggggggggg")

            print("action:", action)
            next_state, reward, done, pos_error = env.step_hover(state_reduced, action) # apply action to drone
            print "03 Pos_error:" + str(pos_error)


            # if pos error decreased, give reward. if increased, give pen
            # if pos_error < prev_pos_error:
            #     reward += 0.2
            # else:
            #     reward -= 0.2

            # reward for being very close to goal
            if pos_error < 0.07**2:
                reward += 0.3

            print("reward", reward)

            prev_pos_error = pos_error # update prev_reward variable

            # print("next state")
            # env.print_state(next_state)

            yaw = next_state[11] # extract next yaw angle

            next_state_reduced = env.reduce_state_array_hover(next_state) # reduce state space to x y z vx vy vz ax ay az

            # subtract goal states from xyz to get relative position to setpoint
            next_state_reduced[0] = next_state_reduced[0] - env.x_goal_hover
            next_state_reduced[1] = next_state_reduced[1] - env.y_goal_hover
            next_state_reduced[2] = next_state_reduced[2] - env.z_goal_hover


            next_state_reduced[0:3] = env.rotate_vector(next_state_reduced[0:3], -yaw) # rotate xyz -yaw degrees about z
            # next_state_reduced[3:6] = env.rotate_vector(next_state_reduced[3:6], -yaw) # rotate xyz -yaw degrees about z
            # next_state_reduced[6:9] = env.rotate_vector(next_state_reduced[6:9], -yaw) # rotate xyz -yaw degrees about z
            next_state_reduced = env.normalize_state_array_hover(next_state_reduced) # normalize to [-1,1]     
            next_state_reduced = np.reshape(next_state_reduced, [1, env.state_size_hover]) # reformat for neat storage


            state = next_state # transfer the state over, but do not overwrite
            total_reward += reward
            steps_completed = step_number
            print "04 prev_pos_error:" + str(prev_pos_error)


        if env.fsm_state == 2:

            yaw = state[11] # extract yaw angle

            # build state vector. reduce state space to x y z vx vy vz ax ay az
            state_reduced = env.reduce_state_array_descend(state)

            # subtract goal states from xyz to get relative position to setpoint
            state_reduced[0] = state_reduced[0] - env.x_goal_descend
            state_reduced[1] = state_reduced[1] - env.y_goal_descend
            state_reduced[2] = state_reduced[2] - env.z_goal_descend
            state_reduced[9] = state_reduced[9] - env.rotz_goal_descend
            
            # rotate pos, vel and acc vectors -yaw degrees about z in state array so they are expressed in body coordinates
            state_reduced[0:3] = env.rotate_vector(state_reduced[0:3], -yaw)

            # normalize each vector inside the state array
            state_reduced = env.normalize_state_array_descend(state_reduced) 

            # reformat for neat storage
            state_reduced = np.reshape(state_reduced, [1, env.state_size_descend]) 

            # choose action
            pure_action = actor_imported_descend.predict(state_reduced)[0]
            action = 0.2 * pure_action
            print("05 prev_pos_error", prev_pos_error)
            if prev_pos_error < 0.05:
                print("DOWNSCALE BENCHOD")
                action = 0.1 * pure_action
                hovered_in_place_counter += 1
                if hovered_in_place_counter >= 12:
                    env.fsm_state = 3
                    print("CHANGED GOAL 2")
                    land()
                    print("land1")
                    

            print("action:", action)
            next_state, reward, done, pos_error, pos_error_aug = env.step_descend(state_reduced, action) # apply action to drone

            print("06 pos_error", pos_error)

            # reward for being very close to goal
            if pos_error < 0.07**2:
                reward += 0.3

            print("reward", reward)

            prev_pos_error = pos_error # update prev_reward variable

            # print("next state")
            # env.print_state(next_state)

            yaw = next_state[11] # extract next yaw angle

            next_state_reduced = env.reduce_state_array_descend(next_state) # reduce state space to x y z vx vy vz ax ay az

            # subtract goal states from xyz to get relative position to setpoint
            next_state_reduced[0] = next_state_reduced[0] - env.x_goal_descend
            next_state_reduced[1] = next_state_reduced[1] - env.y_goal_descend
            next_state_reduced[2] = next_state_reduced[2] - env.z_goal_descend
            next_state_reduced[9] = next_state_reduced[9] - env.rotz_goal_descend


            next_state_reduced[0:3] = env.rotate_vector(next_state_reduced[0:3], -yaw) # rotate xyz -yaw degrees about z
            # next_state_reduced[3:6] = env.rotate_vector(next_state_reduced[3:6], -yaw) # rotate xyz -yaw degrees about z
            # next_state_reduced[6:9] = env.rotate_vector(next_state_reduced[6:9], -yaw) # rotate xyz -yaw degrees about z
            next_state_reduced = env.normalize_state_array_descend(next_state_reduced) # normalize to [-1,1]     
            next_state_reduced = np.reshape(next_state_reduced, [1, env.state_size_descend]) # reformat for neat storage

            state = next_state # transfer the state over, but do not overwrite
            total_reward += reward
            steps_completed = step_number

        while env.fsm_state == 3:
            land()
            print("landing")
            break            


        # stop episode if the drone is out of bounds
        # if done:
        #     break

    return total_reward, steps_completed

        

# if __name__ == '__main__':
while not rospy.is_shutdown():
    # train and obtain (hopefully) optimal q values
    start_of_training = rospy.get_rostime()
    # episode = 0
    for episode in tqdm(range(max_episodes)):
    # while episode < max_episodes:

        if env.fsm_state != 3:
            total_reward, steps_completed = run_episode(max_steps, train_agent)

        # print("")
        # print("episode:", episode)
        # print("reward:", total_reward)
        # print("steps in episode:", steps_completed)
        # print("memory length:", len(env.agent.memory))
        # print("")

        # # if valid amount of steps, we append the episode as a result. bound of if must be equal to bound in run_episode for adding to RB
        # # if steps_completed > 1:
        # #     reward_array.append(total_reward)
        # #     number_of_steps_array.append(steps_completed)

        # reward_array.append(total_reward)
        # number_of_steps_array.append(steps_completed)

        # if not train_agent:
        #     with open('/home/danie/catkin_ws/src/ddpg/src/state_recorder.json', 'w') as fp:
        #         json.dump(list(state_recorder), fp)

        # # store stuff
        # if (episode % 10) == 0 and train_agent: # back up training values
        #     print("storing intermediate results!")
        #     end_of_training = rospy.get_rostime()
        #     training_time = end_of_training.secs - start_of_training.secs
            
        #     with open('/home/danie/catkin_ws/src/ddpg/src/critic_loss_array.json', 'w') as fp:
        #         json.dump(list(np.float64(critic_loss_array)), fp)
        #     with open('/home/danie/catkin_ws/src/ddpg/src/q_value_array.json', 'w') as fp:
        #         json.dump(list(np.float64(q_value_array)), fp)
        #     with open('/home/danie/catkin_ws/src/ddpg/src/training_time.json', 'w') as fp:
        #         json.dump(training_time, fp)
        #     with open('/home/danie/catkin_ws/src/ddpg/src/reward_array.json', 'w') as fp:
        #         json.dump(reward_array, fp)
        #     with open('/home/danie/catkin_ws/src/ddpg/src/number_of_steps_array.json', 'w') as fp:
        #         json.dump(number_of_steps_array, fp)
        #     with open('/home/danie/catkin_ws/src/ddpg/src/replay_buffer.json', 'w') as fp:
        #         json.dump(list(env.agent.memory.memory), fp)
        #     env.agent.actor_local.model.save("/home/danie/catkin_ws/src/ddpg/src/actor_model.hdf5")
        #     env.agent.critic_local.model.save("/home/danie/catkin_ws/src/ddpg/src/critic_model.hdf5")


        # if len(env.agent.memory) > max_total_steps:
        #     print("enough transitions recorded. stop training")
        #     break



    # end_of_training = rospy.get_rostime()
    # training_time = end_of_training.secs - start_of_training.secs
    # with open('/home/danie/catkin_ws/src/ddpg/src/training_time.json', 'w') as fp:
    #     json.dump(training_time, fp)
    # with open('/home/danie/catkin_ws/src/ddpg/src/reward_array.json', 'w') as fp:
    #     json.dump(reward_array, fp)
    # with open('/home/danie/catkin_ws/src/ddpg/src/replay_buffer.json', 'w') as fp:
    #     json.dump(list(env.agent.memory.memory), fp)
    # env.agent.actor_local.model.save("/home/danie/catkin_ws/src/ddpg/src/actor_model.hdf5")
    # env.agent.critic_local.model.save("/home/danie/catkin_ws/src/ddpg/src/critic_model.hdf5")
    # reason = "episodes done alhamdulillah!!"
    # rospy.signal_shutdown(reason)
    break
    rate.sleep() # do i need this?