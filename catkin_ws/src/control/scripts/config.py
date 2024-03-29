#!/usr/bin/env python
"""
Config settings and parameters for control package
"""

import numpy as np
import math

# General
is_simulator = False

default_setpoint = np.array([0,0,1.75, 0, 0, 0])

takeoff_height = 1.75

##########################
#  MISSION PARAMETERS    #
##########################
close_enough_euc = 0.5
close_enough_ang = 175
hover_duration = 5
landing_timer_duration = 10
takeoff_timer_duration = 3
slice_on = not is_simulator
slice_duration_move = 2
slice_duration_stop = 2

# ABORT MISSION PARAMETERS
x_upper_limit = 15 if not is_simulator else 15000
y_upper_limit = 15 if not is_simulator else 15000
z_upper_limit = 7 if not is_simulator else 15000
error_timer_duration = 500 if not is_simulator else 500
error_descent_vel = -0.4




####################
#  PID parameters  #
####################

actuation_saturation = 1 # % of maximum velocity
error_integral_limit = 40 if is_simulator else 20


if is_simulator:
    Kp_position_x = 0.7
    Ki_position_x = 0.01
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
    Kp_orientation = 0.02
    Ki_orientation = 0.0
    Kd_orientation = 0.0
    ####################

else:
    Kp_position_x = 0.10
    Ki_position_x = 0.001
    Kd_position_x = 0.00
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
