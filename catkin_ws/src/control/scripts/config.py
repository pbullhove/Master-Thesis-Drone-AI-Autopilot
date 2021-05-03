#!/usr/bin/env python
import numpy as np
import math

# General
is_simulator = True

# Dead reckoning
do_calibration_before_start = False


reference_height = 1.0
# delta_x, delta_y = height_to_delta_x_y(reference_height)
delta_x, delta_y = 0,0

offset_setpoint_x = delta_x             # 0.79 (at h = 2.0) # 1.77 (at h = 4.0)
offset_setpoint_y = delta_y                     # 1.41              # 3.15
controller_desired_pose = np.array([offset_setpoint_x, offset_setpoint_y, reference_height, 0.0, 0.0, 0.0])



##########################
#  MISSION PARAMETERS    #
##########################
close_enough_euc = 0.1
close_enough_ang = 3
error_timeout_time = 100
hover_duration = 5
error_descent_vel = -0.4
takeoff_height = 3


####################
#  PID parameters  #
####################

actuation_saturation = 1 # % of maximum velocity
error_integral_limit = 40


if is_simulator:
    Kp_position_x = 0.7
    Ki_position_x = 0.001
    Kd_position_x = 0.5
    ####################
    Kp_position_y = Kp_position_x
    Ki_position_y = Ki_position_x
    Kd_position_y = Kd_position_x
    ####################
    Kp_position_z = 0.7
    Ki_position_z = 0.001
    Kd_position_z = 0.5
    ####################
    Kp_orientation = 0.01
    Ki_orientation = 0.0
    Kd_orientation = 0.0
    ####################

else:
    Kp_position_x = 1.5
    Ki_position_x = 0.01
    Kd_position_x = 0.3
    ####################
    Kp_position_y = Kp_position_x
    Ki_position_y = Ki_position_x
    Kd_position_y = Kd_position_x
    ####################
    Kp_position_z = 1.5
    Ki_position_z = 0.01
    Kd_position_z = 0.3
    ####################
    Kp_orientation = 0.0
    Ki_orientation = 0.0
    Kd_orientation = 0.0
    ####################
