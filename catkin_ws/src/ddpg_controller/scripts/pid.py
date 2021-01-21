import numpy as np
import rospy


class PID:
    """PID Controller"""

    def __init__(self, params_x, params_y, params_z, params_rotz, frequency=10, current_time=None):

        self.Kp_x = params_x[0]
        self.Ki_x = params_x[1]
        self.Kd_x = params_x[2]

        self.Kp_y = params_y[0]
        self.Ki_y = params_y[1]
        self.Kd_y = params_y[2]

        self.Kp_z = params_z[0]
        self.Ki_z = params_z[1]
        self.Kd_z = params_z[2]

        self.Kp_rotz = params_rotz[0]
        self.Ki_rotz = params_rotz[1]
        self.Kd_rotz = params_rotz[2]
        

        # inverse of frequency is the sample time
        self.sample_time = 1.0/frequency
        # self.sample_time = rospy.Duration(0.00833)


        # to keep track of how often to update the control
        # self.current_time = current_time if current_time is not None else time.time()
        self.current_time = current_time if current_time is not None else rospy.get_time()
        self.last_time = self.current_time
        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        # K_p * e
        self.PTerm_x = 0.0
        self.PTerm_y = 0.0
        self.PTerm_z = 0.0
        self.PTerm_rot_z = 0.0
        # K_i * integral(e)
        self.ITerm_x = 0.0
        self.ITerm_y = 0.0
        self.ITerm_z = 0.0
        self.ITerm_rot_z = 0.0

        # K_d * derivative(e)
        self.DTerm_x = 0.0
        self.DTerm_y = 0.0
        self.DTerm_z = 0.0
        self.DTerm_rot_z = 0.0
        self.last_error_x = 0.0
        self.last_error_y = 0.0
        self.last_error_z = 0.0
        self.last_error_rot_z = 0.0
     
        # Windup Guard for integrator
        self.windup_guard = 0.5

        self.output_x = 0.0
        self.output_y = 0.0
        self.output_z = 0.0
        self.output_rot_z = 0.0

    def run(self, curr_x, goal_x, curr_y, goal_y, curr_z, goal_z, curr_rot_z, goal_rot_z, current_time=None):
        """
            Calculates PID value for given reference feedback
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        
        error_x = goal_x - curr_x
        error_y = goal_y - curr_y
        error_z = goal_z - curr_z
        error_rot_z = goal_rot_z - curr_rot_z

        self.output_x = 0.0
        self.output_y = 0.0
        self.output_z = 0.0
        self.output_rot_z = 0.0

        # self.current_time = current_time if current_time is not None else rospy.get_rostime()
        # self.current_time = time.time()
        self.current_time = rospy.get_time()
        delta_time = self.current_time - self.last_time
        delta_error_x = error_x - self.last_error_x
        delta_error_y = error_y - self.last_error_y
        delta_error_z = error_z - self.last_error_z
        delta_error_rot_z = error_rot_z - self.last_error_rot_z

        # print("delta_time:", delta_time)

        if (delta_time >= self.sample_time):
            self.PTerm_x = self.Kp_x * error_x
            self.PTerm_y = self.Kp_y * error_y
            self.PTerm_z = self.Kp_z * error_z
            self.PTerm_rot_z = self.Kp_rotz * error_rot_z

            self.ITerm_x += error_x * delta_time
            self.ITerm_y += error_y * delta_time
            self.ITerm_z += error_z * delta_time
            self.ITerm_rot_z += error_rot_z * delta_time

            if (self.ITerm_x < -self.windup_guard):
                self.ITerm_x = -self.windup_guard
            elif (self.ITerm_x > self.windup_guard):
                self.ITerm_x = self.windup_guard
            
            if (self.ITerm_y < -self.windup_guard):
                self.ITerm_y = -self.windup_guard
            elif (self.ITerm_y > self.windup_guard):
                self.ITerm_y = self.windup_guard

            if (self.ITerm_z < -self.windup_guard):
                self.ITerm_z = -self.windup_guard
            elif (self.ITerm_z > self.windup_guard):
                self.ITerm_z = self.windup_guard
            
            if (self.ITerm_rot_z < -self.windup_guard):
                self.ITerm_rot_z = -self.windup_guard
            elif (self.ITerm_rot_z > self.windup_guard):
                self.ITerm_rot_z = self.windup_guard

            self.DTerm_x = 0.0
            self.DTerm_y = 0.0
            self.DTerm_z = 0.0
            self.DTerm_rot_z = 0.0
            if delta_time > 0:
                self.DTerm_x = delta_error_x / delta_time
                self.DTerm_y = delta_error_y / delta_time
                self.DTerm_z = delta_error_z / delta_time
                self.DTerm_rot_z = delta_error_rot_z / delta_time


            # Remember last time and last error for next calculation
            self.last_time = self.current_time
            self.last_error_x = error_x
            self.last_error_y = error_y
            self.last_error_z = error_z
            self.last_error_rot_z = error_rot_z

            self.output_x  = self.PTerm_x + (self.Ki_x * self.ITerm_x) + (self.Kd_x * self.DTerm_x)
            self.output_y  = self.PTerm_y + (self.Ki_y * self.ITerm_y) + (self.Kd_y * self.DTerm_y)
            self.output_z  = self.PTerm_z + (self.Ki_z * self.ITerm_z) + (self.Kd_z * self.DTerm_z)
            self.output_rot_z = self.PTerm_rot_z + (self.Ki_rotz * self.ITerm_rot_z) + (self.Kd_rotz * self.DTerm_rot_z)
        
        return self.output_x, self.output_y, self.output_z, self.output_rot_z
