#!/usr/bin/env python
"""
Config settings and parameters for control package
"""

import numpy as np
import math

# General
is_simulator = True

default_setpoint = np.array([-0.1, 0, 1.5, 0, 0, 0])

takeoff_height = 1.5

##########################
#  MISSION PARAMETERS    #
##########################
close_enough_euc = 0.1
close_enough_ang = 3
hover_duration = 5
landing_timer_duration = 10
takeoff_timer_duration = 3

# ABORT MISSION PARAMETERS
x_upper_limit = 15
y_upper_limit = 15
z_upper_limit = 7
error_timer_duration = 90
error_descent_vel = -0.4

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
    Kp_position_x = 0.10
    Ki_position_x = 0.0
    Kd_position_x = 0.5
    ####################
    Kp_position_y = Kp_position_x
    Ki_position_y = Ki_position_x
    Kd_position_y = Kd_position_x
    ####################
    Kp_position_z = 0.10
    Ki_position_z = 0.001
    Kd_position_z = 0.1
    ####################
    Kp_orientation = 0.0
    Ki_orientation = 0.0
    Kd_orientation = 0.0
    ####################
