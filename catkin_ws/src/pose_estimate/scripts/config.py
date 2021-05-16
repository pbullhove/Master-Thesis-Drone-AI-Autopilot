#!/usr/bin/env python

"""
Config settings for pose_estimate package
"""
import numpy as np
import math

# General
is_simulator = False
do_calibration_before_start = True
num_calib_steps = 1000
vel_estimate_limit = 0.7

# CV module
save_images = False
draw_on_images = False
use_test_image = False
publish_processed_image = False
