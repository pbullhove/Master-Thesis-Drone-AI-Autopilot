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
from collections import deque
import tensorflow as tf

# ros oriented packages 
import roslib
import rospy
from std_srvs.srv import Empty as empty
from std_msgs.msg import Empty
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

## Initialize node, publishers and services
rospy.init_node('control_script')
rate_hz = 20
rate = rospy.Rate(rate_hz)

pub_takeoff = rospy.Publisher('/ardrone/takeoff', Empty, queue_size=1000)
# pub_land = rospy.Publisher('/ardrone/land', Empty, queue_size=1000)
pub_action = rospy.Publisher('/cmd_vel', Twist, queue_size=1000)

reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', empty)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', empty)

# Define the system: behavior and helping functions
class DroneState():
    # Initializes internal variables
    def __init__(self):
        # get size of state and action
        self.state_size = 12 # x y z vx vy vz rotx roty rotz a_prev
        self.action_size = 3 # velocity in x y z

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
        self.a_prev = [0.0] * self.action_size

        # state sub arrays
        self.drone_state_odometry = [self.x, self.y, self.z] # 3
        self.drone_state_navdata = [self.vx, self.vy, self.vz, self.ax, self.ay, self.az, self.rot_x, self.rot_y, self.rot_z] # 9
        self.drone_state_imu = [self.wx, self.wy, self.wz] # 3
        self.drone_state_previous_action = self.a_prev # 3 * self.action_size
        
        # complete state array
        # self.drone_state = [self.x, self.y, self.z, self.vx, self.vy, self.vz, self.ax, \
        #     self.ay, self.az, self.rot_x, self.rot_y, self.rot_z, self.wx, self.wy, self.wz, self.a_curr, self.a_prev]

        self.drone_state = self.drone_state_odometry + self.drone_state_navdata + self.drone_state_imu + self.drone_state_previous_action

        # saturation of actions
        self.action_low = -0.4 # set to the same as Lasse did
        self.action_high = 0.4 # set to the same as Lasse did

        # set the bounds of x, y, z and rotz for. showcasing the state space the drone can find itself in in the environment
        self.minx = -2.0
        self.maxx =  2.0
        self.miny = -2.0
        self.maxy =  2.0
        self.minz = 1.0
        self.maxz = 3.0
        self.minrotx = -20
        self.maxrotx = 20
        self.minroty = -20
        self.maxroty = 20
        self.minrotz = -90
        self.maxrotz = 90
        # self.minrotz = -20.0
        # self.maxrotz = 20.0

        self.agent = DDPGAgent(self.state_size, self.action_size, self.action_low, self.action_high)

    # Reduces the original state vector. Only preserves states that we actually use
    def reduce_state_array(self, state):
        # x y z vx vy vz rotX rotY rotZ a_prev
        return [state[0], state[1], state[2], state[3], state[4], state[5], state[9], state[10], state[11], state[15], state[16], state[17]]

    # https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    # normalizes a (state) s, that originally has bounds [mins, maxs] and returns x with bounds [a,b]
    def normalize_state(self, s, mins, maxs, a, b):
        return (b - a) * (s - mins) / (maxs - mins) + a
        # if isinstance(s, float):
        #     return (b - a) * (s - mins) / (maxs - mins) + a
        # elif isinstance(s, list):
        #     output_array = [0.0]*len(s) # initialize array of same size as input array
        #     for i in range(len(s)):
        #         output_array[i] = (b - a) * (s[i] - mins) / (maxs - mins) + a
        #     return output_array
        # else:
        #     print("Faulty input to normalize_state")
        #     return -1


    # normalize x, y, z, rotz state array to range of [1,1]
    def normalize_state_array(self, state_reduced):
        norm_list = []
        # normalize x y z according to boundaries set in the environment
        norm_x = self.normalize_state(state_reduced[0], self.minx, self.maxx, -1, 1) 
        norm_y = self.normalize_state(state_reduced[1], self.miny, self.maxy, -1, 1)
        norm_z = self.normalize_state(state_reduced[2], self.minz, self.maxz, -1, 1)
        norm_list.append(norm_x)
        norm_list.append(norm_y)
        norm_list.append(norm_z)

        # normalize velocities according to the action thresholds (remember that actions are velocities!)
        # min and max for velocities come from testing. See log 24.2
        norm_vx = self.normalize_state(state_reduced[3], -1100, 1100, -1, 1) 
        norm_vy = self.normalize_state(state_reduced[4], -1200, 1200, -1, 1)
        norm_vz = self.normalize_state(state_reduced[5], -600, 600, -1, 1)
        norm_list.append(norm_vx)
        norm_list.append(norm_vy)
        norm_list.append(norm_vz)

        # normalize angles according to predefined specs
        norm_rotx = self.normalize_state(state_reduced[6], self.minrotx, self.maxrotx, -1, 1)
        norm_roty = self.normalize_state(state_reduced[7], self.minroty, self.maxroty, -1, 1)
        norm_rotz = self.normalize_state(state_reduced[8], self.minrotz, self.maxrotz, -1, 1)
        norm_list.append(norm_rotx)
        norm_list.append(norm_roty)
        norm_list.append(norm_rotz)

        # normalize actions according to the action thresholds. Code should be flexible to changes in action_size. Code assumes similar action boundaries for all actions
        for i in range(self.action_size):
            norm_a_prev = self.normalize_state(state_reduced[9+i], self.action_low, self.action_high, -1, 1)
            norm_list.append(norm_a_prev)

        return norm_list


    # Sets x, y and z
    def set_drone_state_odometry(self, state_odometry):
        self.drone_state_odometry = state_odometry
        # self.drone_state_odometry = [state_odometry[0], state_odometry[1], state_odometry[2]]
    
    # Sets vx, vy, vz, ax, ay, az, rotx, roty, rotz
    def set_drone_state_navdata(self, state_navdata):
        self.drone_state_navdata = state_navdata
        # self.drone_state_navdata = [state_navdata[0], state_navdata[1], state_navdata[2]]

    # Sets wx, wy, wz
    def set_drone_state_imu(self, state_imu):
        self.drone_state_imu = state_imu

    def set_drone_state_previous_action(self, a_prev):
        self.drone_state_previous_action = a_prev

    # Sets x, y, z, vx, vy, vz, ax, ay, az, rotx, roty, rotz, wx, wy, wz, a_prev
    def set_drone_state(self, state_odometry, state_navdata, state_imu, a_prev):
        self.set_drone_state_odometry(state_odometry)
        self.set_drone_state_navdata(state_navdata)
        self.set_drone_state_imu(state_imu)
        self.set_drone_state_previous_action(a_prev)

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
        state_navdata = self.get_drone_state_navdata()
        state_imu = self.get_drone_state_imu()
        a_prev = self.get_drone_state_previous_action()
        return state_odometry + state_navdata + state_imu + a_prev

    # Checks if the drone is not in a valid area
    def out_of_bounds(self, state):
        is_out_of_bounds = False
        # if state[0] < self.minx or state[0] > self.maxx:
        #     print("out of bounds in x")
        #     is_out_of_bounds = True
        # if state[1] < self.miny or state[1] > self.maxy:
        #     print("out of bounds in y")
        #     is_out_of_bounds = True
        if state[2] < self.minz or state[2] > self.maxz:
            print("out of bounds in z")
            is_out_of_bounds = True

        return is_out_of_bounds

    # Resets the world
    def reset(self):
        rospy.wait_for_service('gazebo/reset_world')
        try:
            reset_simulation()
        except(rospy.ServiceException) as e:
            print("reset_world failed!")
        print("\n called reset()")

        self.set_drone_state([0.0]*3, [0.0]*9, [0.0]*3, [0.0]*self.action_size)
        return self.get_drone_state()
    
    # Transports the drone to a random start state within the valid area
    def set_start_state(self):
        start_position = [random.uniform(env.minx, env.maxx), random.uniform(env.miny, env.maxy), random.uniform(env.minz, env.maxz)]
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

        self.set_drone_state(start_position, [0.0]*9, [0.0]*3, [0.0]*self.action_size)
        return self.get_drone_state()


    # Applies an action to the drone and returns the next state, the reward and if the episode is done
    def step(self, action):

        command = Twist()
        
        # get action from actor
        u_x, u_y, u_z  = action[0], action[1], action[2]
        command.linear.x = u_x
        command.linear.y = u_y
        command.linear.z = u_z
        command.angular.x = 0.0
        command.angular.y = 0.0
        command.angular.z = 0.0
        pub_action.publish(command)

        rate.sleep() # allow the drone some time before it enters the next state

        # get the new state we have arrived at AFTER action is done, and get the reward
        next_state = self.get_drone_state()
        reward, done = self.get_reward(next_state, action)

        # return next_state, reward, env.done
        
        return next_state, reward, done
    
    # reward function that penalizes the drone for not hovering at z=2
    def get_reward(self, state, current_action):
        reward = 0.0
        # want to hover at z=2
        goal_x = 0.0
        goal_y = 0.0
        goal_z = 2.0

        # flag indicating that the episode is done
        done = False

        delta_x = state[0] - goal_x
        delta_y = state[1] - goal_y
        delta_z = state[2] - goal_z

        factor_delta_action = 0.05
        delta_action = np.array(current_action) - np.array(self.get_drone_state_previous_action())

        factor_action = 0.02
        current_action = np.array(current_action)

        # reward = 1 - 1 * ( delta_x**2 + delta_y**2 + delta_z**2 )

        # lasse's reward
        sigma = 0.05
        # print("exp pen", np.exp( - (delta_x**2 + delta_y**2 + delta_z**2) / (2 * sigma) ))
        # print("delta pen", -factor_delta_action * np.linalg.norm(delta_action))
        # print("action pen", -factor_action * np.linalg.norm(current_action))
        # reward = np.exp( - (delta_x**2 + delta_y**2 + delta_z**2) / (2 * sigma) ) - factor_delta_action * np.linalg.norm(delta_action) - factor_action * np.linalg.norm(current_action)
        reward = np.exp( - (delta_x**2 + delta_y**2 + delta_z**2) / (2 * sigma) )

        if self.out_of_bounds(state):
            done = True

        return reward, done


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
# def land():
#     pub_land.publish(Empty())

# Callback: setting x y z
# http://docs.ros.org/melodic/api/nav_msgs/html/msg/Odometry.html
def cb_drone_state_odometry(data):
    env.set_drone_state_odometry([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])

def cb_drone_state_point_thomas(data):
    env.set_drone_state_odometry([data.x, data.y, data.z])

# Callback: setting vx vy vz ax ay az rotx roty rotz
# http://docs.ros.org/indigo/api/ardrone_autonomy/html/msg/Navdata.html
def cb_drone_state_navdata(data):
    env.set_drone_state_navdata([data.vx, data.vy, data.vz, data.ax, data.ay, data.az, data.rotX, data.rotY, data.rotZ])

# Callback: setting wx wy wz
# http://docs.ros.org/api/sensor_msgs/html/msg/Imu.html
def cb_drone_state_imu(data):
    env.set_drone_state_imu([data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z])

# Makes the subscribers
def subscriber():
    rospy.Subscriber("/ground_truth/state", Odometry, cb_drone_state_odometry)
    # rospy.Subscriber("/drone_estimate", Point, cb_drone_state_point_thomas)
    rospy.Subscriber("/ardrone/navdata", Navdata, cb_drone_state_navdata)
    rospy.Subscriber("/ardrone/imu", Imu, cb_drone_state_imu)


num_episodes = 50000 # number of episodes to train drone in
max_steps = 50000 # max number of steps in total for the whole training period
exploration_steps = 1 # number of steps agent should explore before going to normal ddpg action choice

subscriber()

reward_array = np.zeros(num_episodes)

train_agent = False

if train_agent == False:
    actor_imported = tf.keras.models.load_model('/home/danie/catkin_ws/src/ddpg/src/results/mar2/actor_model.hdf5')


if train_agent:
    # train and obtain (hopefully) optimal q values
    while not rospy.is_shutdown():
        start_of_training = rospy.get_rostime()
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            takeoff()
            # rospy.sleep(0.1)
            
            # choose start coordinate of the episode
            if episode == 0:
                start_x = 0.0
                start_y = 0.0
                start_z = 1.0
            else:
                # start_x = random.uniform(env.minx, env.maxx)
                # start_y = random.uniform(env.miny, env.maxy)
                start_x = random.uniform(env.minx, env.maxx)
                start_y = random.uniform(env.miny, env.maxy)
                start_z = random.uniform(env.minz, env.maxz)

            print("start state is: ({},{},{})".format(start_x, start_y, start_z))
            
            # drive the drone to starting position using PID
            state = env.set_start_state()
            

            # Now that drone is in position, start training with ddpg
            done = False
            total_reward = 0
            time_start_of_episode = rospy.get_rostime()
            now = rospy.get_rostime()
            step_number = 0
            print("start episode")
            state = env.get_drone_state()
            while (now.secs - time_start_of_episode.secs) <= 10:
                step_number += 1
                now = rospy.get_rostime()

                # build state vector
                state_reduced = env.reduce_state_array(state) # reduce state space to x y z rotz
                # print("x,y,z", state[0], state[1], state[2])
                state_reduced = env.normalize_state_array(state_reduced) # normalize to [-1,1]
                state_reduced = np.reshape(state_reduced, [1, env.state_size]) # reformat for neat storage

                # choose action
                if len(env.agent.memory) < exploration_steps:
                    # print("random action")
                    action = np.random.uniform(env.action_low, env.action_high, env.action_size).tolist()
                else:
                    action = env.agent.choose_action(state_reduced) # state --> action
                
                next_state, reward, done = env.step(action) # apply action to drone

                env.set_drone_state_previous_action(action) # store action in drone state AFTER step is made to ensure a_prev and a_curr is not the same in get_reward

                next_state_reduced = env.reduce_state_array(next_state) # reduce state space to x y z rotz
                next_state_reduced = env.normalize_state_array(next_state_reduced) # normalize to [-1,1]           
                next_state_reduced = np.reshape(next_state_reduced, [1, env.state_size]) # reformat for neat storage

                env.agent.store_transition(state_reduced, action, reward, next_state_reduced, done)

                env.agent.train_actor_and_critic()

                state = next_state # dont really need this line since i call get_drone_state when the loop goes back to the top

                total_reward += reward
                
                # stop episode if the drone has landed/out of bounds or if it takes too long. 
                if done:
                    break

                rate.sleep()
            
            if len(env.agent.memory) > max_steps:
                print("enough transitions recorded. stop training")
                break

            print("episode:", episode, "  reward:", total_reward, "  memory length:", len(env.agent.memory))
            reward_array[episode] = total_reward

            if (episode % 10) == 0 and episode != 0: # back up training values
                print("storing intermediate results!")
                end_of_training = rospy.get_rostime()
                training_time = end_of_training.secs - start_of_training.secs
                with open('/home/danie/catkin_ws/src/ddpg/src/training_time.json', 'w') as fp:
                    json.dump(training_time, fp)
                with open('/home/danie/catkin_ws/src/ddpg/src/reward_array.json', 'w') as fp:
                    json.dump(reward_array.tolist(), fp)
                env.agent.actor_local.model.save("/home/danie/catkin_ws/src/ddpg/src/actor_model.hdf5")
                env.agent.critic_local.model.save("/home/danie/catkin_ws/src/ddpg/src/critic_model.hdf5")


        end_of_training = rospy.get_rostime()
        training_time = end_of_training.secs - start_of_training.secs
        with open('/home/danie/catkin_ws/src/ddpg/src/training_time.json', 'w') as fp:
            json.dump(training_time, fp)
        with open('/home/danie/catkin_ws/src/ddpg/src/reward_array.json', 'w') as fp:
            json.dump(reward_array.tolist(), fp)
        env.agent.actor_local.model.save("/home/danie/catkin_ws/src/ddpg/src/actor_model.hdf5")
        env.agent.critic_local.model.save("/home/danie/catkin_ws/src/ddpg/src/critic_model.hdf5")
        reason = "episodes done alhamdulillah!!"
        rospy.signal_shutdown(reason)
else:

    while not rospy.is_shutdown():
        start_of_training = rospy.get_rostime()
        for episode in tqdm(range(num_episodes)):
            state = env.reset()
            takeoff()
            # rospy.sleep(0.1)
            
            # choose start coordinate of the episode
            if episode == 0:
                start_x = 0.0
                start_y = 0.0
                start_z = 1.0
            else:
                start_x = random.uniform(env.minx, env.maxx)
                start_y = random.uniform(env.miny, env.maxy)
                start_z = random.uniform(env.minz, env.maxz)

            print("start state is: ({},{},{})".format(start_x, start_y, start_z))
            
            # drive the drone to starting position using PID
            state = env.set_start_state()

            # Now that drone is in position, start training with ddpg
            done = False
            total_reward = 0
            time_start_of_episode = rospy.get_rostime()
            now = rospy.get_rostime()
            step_number = 0
            while (now.secs - time_start_of_episode.secs) <= 50:
                step_number += 1
                now = rospy.get_rostime()

                # build state vector
                state_reduced = env.reduce_state_array(state) # reduce state space to x y z rotz
                print("x,y,z", state[0], state[1], state[2])
                state_reduced = env.normalize_state_array(state_reduced) # normalize to [-1,1]
                state_reduced = np.reshape(state_reduced, [1, env.state_size]) # reformat for neat storage
                
                 # feed state and get action. didnt bother making a function
                pure_action = actor_imported.predict(state_reduced)[0]
                # print("pure action", pure_action)
                action = pure_action * env.action_high

                next_state, reward, done = env.step(action) # apply action to drone

                env.set_drone_state_previous_action(action.tolist()) # store action in drone state AFTER step is made to ensure a_prev and a_curr is not the same in get_reward

                next_state_reduced = env.reduce_state_array(next_state) # reduce state space to x y z rotz
                next_state_reduced = env.normalize_state_array(next_state_reduced) # normalize to [-1,1]           
                next_state_reduced = np.reshape(next_state_reduced, [1, env.state_size]) # reformat for neat storage

                state = next_state # dont really need this line since i call get_drone_state when the loop goes back to the top

                total_reward += reward
                
                # if (now.secs - time_start_of_episode.secs) >= 75:
                #     total_reward -= 100
                #     print("drone took too long landing")
                #     break

                # stop episode if the drone has landed/out of bounds or if it takes too long. 
                if done:
                    break

                rate.sleep()
            
            print("episode:", episode, "  reward:", total_reward, "  memory length:", len(env.agent.memory))
            reward_array[episode] = total_reward