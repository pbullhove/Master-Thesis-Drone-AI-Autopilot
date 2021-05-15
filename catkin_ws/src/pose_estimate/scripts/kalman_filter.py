#!/usr/bin/env python
"""
A pose estimation Kalman Filter which combines data from mulitiple sensors in a statistically
optimal way, based on estimated sensor noise. Predictions are performed using IMU data velocities and accelerations.
Updates are performed with the other sensors.

x_est = (x, y, z, roll, pitch, yaw)
roll, pitch := (0,0)
P = error covariance matrix, meaning how uncertain is the current x_est.

Sensors:
- Camera: (two methods, dnnCV, and tcv)
- IMU: drone velocities and accelerations
- Sonar: drone altitude
- Mock gps: in the simulated environmet, not available on real ardrone.

Tune Kalman Filter by changing R-values for the desired sensors, where R specifies measurement noise,
and by changing Q_imu which specifies how uncertain IMU predictions are.

Subscribes to:
    /estimate/dnnCV: Twist - Pose estimates from the dnn CV method
    /estimate/tcv: Twist - Pose estimates from the tcv CV method.
    /mock_gps: Twist - Pose estimates from the simulator mock gps.
    /ardrone/navdata: Odometry - Odometry data from ardrone.

Publishes to:
    /filtered_estimate: Twist - the current estimated quadcopter pose
"""
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import time
import math
import config as cfg
import pe_help_functions as hlp

x_est = np.zeros(6)
x_est_prev = np.zeros(6)
v_est = np.zeros(3)
P = np.eye(6)
P_v = np.eye(3)
prev_KFU_t = 0

C_yolo = np.eye(6)
C_tcv = np.eye(6)
C_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
C_sonar = np.array([0,0,1,0,0,0]).reshape((1,6))
C_baro = np.array([0,0,1,0,0,0]).reshape((1,6))

R_yolo = 0.1*np.eye(6)
R_tcv = 0.1*np.eye(6)
R_gps = 0.1*np.eye(3)
R_sonar = 0.1*np.eye(1)
R_baro = 3*np.eye(1)

Q_imu = 1*np.diag([1, 1, 1, 0, 0, 1])

calibration_vel = np.array([0.0, 0.0, 0.0])
calibration_acc = np.array([0.0, 0.0, 0.0])
calibration_pressure = 0.0
calib_pitch = 0.0
calib_roll = 0.0
calib_steps = 0
low_vel_limit = 0.003
low_acc_limit = 0.1
ONE_g = 9.8067
vel_decay = 0.0
prev_imu_yaw = None
prev_navdata_timestamp = None

t_prev_range = None
t_prev_gps = None
t_prev_dnn = None
t_prev_tcv = None
y_prev_range = None
y_prev_gps = None
y_prev_dnn = None
y_prev_tcv = None

def kalman_gain(P_k,C,R):
    """ Computes the Kalman Gain which specifies how much to update x_est given a new measurement. """
    PCT = np.dot(P_k, C.T)
    IS = R + np.dot(C, PCT)
    IS_inv = np.linalg.inv(IS)
    K = np.dot(PCT,IS_inv)
    return K

def P_post(P, C, K):
    """ Updating P after updateing x_est on new data"""
    P = np.dot((np.eye(6) - np.dot(K,C)), P)
    return P

def P_apri(P, Q):
    """ Increaing P when predicting.
    P = FPF^T + Q. No F is availabe, so simplified implementation."""
    P = P + Q
    return P

def KF_update(R,C,y):
    """ Updates x_est and P given sensor measurement y, with sensor noise R, and sensor matrix C.
    Predicts yaw within [-180, 180] range.

    output:
        modifies P, x_est
    """
    global P
    global x_est
    global x_est_prev
    global v_est
    K = kalman_gain(P, C, R)
    innov = y-np.dot(C,x_est)
    try: # for jumping between yaw = 179, -179
        innov[5] = hlp.angleFromTo(innov[5], -180, 180)
    except IndexError as e:
        pass
    update = np.dot(K, innov)
    x_est = x_est + update
    x_est[5] = hlp.angleFromTo(x_est[5], -180, 180)
    P = P_post(P,C,K)

def KF_vel_update(R, C, y, sensor_name):
    global v_est
    global P_v
    if sensor_name == "range":
        t_prev =
    try:
        delta_t = datetime.now() - t_prev
        prev_KFU_t = datetime.now()
    except TypeError as e:
        return
    y_vel = (y - y_prev)/delta_t
    K = kalman_gain(P_v, C, R)
    vel_innov = y_vel - np.dot(C[:,0:3], v_est)
    v_est = v_est + np.dot(K, vel_innov)
    P_v = P_post(P,C,K)


def yolo_estimate_callback(data):
    """ Filters pose estimates from yolo cv algorithm. Estimates pos in xyz and yaw. Only use this if more than 0.7m above platform, as camera view too close for correct estimates. """
    global x_est
    global P
    yolo_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.7 or yolo_estimate[2] > 0.7:
        if yolo_estimate[5] == 0.0: #if no estimate for yaw
            C = C_gps
            y = yolo_estimate[0:3]
            R = R_yolo[0:3,0:3]
        else:
            C = C_yolo
            R = R_yolo
            y = yolo_estimate
        KF_update(R,C,y)


def tcv_estimate_callback(data):
    """ Filters pose estimates from tcv cv algorithm. Estimates pos in xyz and yaw. Only use this if mmore than 0.4m above platform, as camera view too close for correct estimates. """
    tcv_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.4 or tcv_estimate[2] > 0.4:
        if tcv_estimate[5] == 0.0 or tcv_estimate[5] == -0.0: #if no estimate for yaw
            C = C_gps
            y = tcv_estimate[0:3]
            R = R_tcv[0:3,0:3]
        else:
            C = C_tcv
            R = R_tcv
            y = tcv_estimate
        KF_update(R,C,y)


def gps_callback(data):
    """ Filters gps data which is measurement of position in xyz. """
    gps_measurement = hlp.twist_to_array(data)
    y = gps_measurement[0:3]
    KF_update(R_gps, C_gps, y)



def navdata_callback(data):
    """
    Filters estimates from the IMU data and predicts quadcopter pose.
    Updates estimate with barometric pressure data.

    Performs calibration before start by setting calibration_vel, calibration_acc to average of standstill values for about 5 seconds.
    Does not publish any estimates before after calibration. Toggle calibration in config.

    input:
        data: Odometry from ardrone 200hz
    output:    global x_est_prev
        changes x_est, P
    """
    global x_est
    global v_est
    global P
    global Q_imu
    global prev_imu_yaw
    global calib_steps
    global prev_navdata_timestamp
    global calibration_vel
    global calibration_acc
    global calibration_pressure
    global calib_roll
    global calib_pitch
    if cfg.do_calibration_before_start and calib_steps < cfg.num_calib_steps:
        calibration_vel += np.array([data.vx, data.vy, data.vz])/float(cfg.num_calib_steps)
        calibration_acc += np.array([data.ax, data.ay, data.az - 1])*ONE_g/float(cfg.num_calib_steps)
        calibration_pressure += data.pressure/float(cfg.num_calib_steps)
        calib_roll += data.rotX/float(cfg.num_calib_steps)
        calib_pitch += data.rotY/float(cfg.num_calib_steps)
        calib_steps += 1
        return

    try:
        delta_yaw = data.rotZ - prev_imu_yaw
        delta_yaw = hlp.angleFromTo(delta_yaw, -180,180)
        now = datetime.now()
        delta_t = (now - prev_navdata_timestamp).total_seconds()
        prev_navdata_timestamp = now
    except TypeError as e: #first iteration
        delta_yaw = 0
        delta_t = 0
        prev_navdata_timestamp = datetime.now()
    finally:
        prev_imu_yaw = data.rotZ

    vel = (np.array([data.vx, data.vy, data.vz]) - calibration_vel)/1000
    acc = np.array([data.ax*ONE_g, data.ay*ONE_g, data.az*ONE_g]) - calibration_acc
    p = hlp.deg2rad(data.rotY - calib_pitch)
    r = hlp.deg2rad(data.rotX - calib_roll)
    gravity_vec = ONE_g*np.array([-math.sin(p),math.cos(p)*math.sin(r), math.cos(p)*math.cos(r)])
    acc -= gravity_vec
    extreme_values_filter_val = np.logical_and(np.less(vel, low_vel_limit), np.greater(vel, -low_vel_limit))
    extreme_values_filter_acc = np.logical_and(np.less(acc, low_acc_limit), np.greater(acc, -low_acc_limit))
    vel[extreme_values_filter_val] = 0.0
    acc[extreme_values_filter_acc] = 0.0


    v_est *= 1-vel_decay
    v_est += delta_t*acc
    v_est = np.array([max(-0.7,i) for i in v_est])
    v_est = np.array([min(0.7,i) for i in v_est])
    delta_pos = delta_t*v_est + 0.5*acc*delta_t**2

    delta_x = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, delta_yaw])
    x_est = x_est + delta_x

    r = R.from_euler('z', -np.radians(delta_yaw))
    x_est[0:3] = r.apply(x_est[0:3])
    P = P_apri(P, Q_imu)
    x_est[5] = hlp.angleFromTo(x_est[5],-180,180)

    y = data.pressure # Z measurement for real QC
    if y > 0.0:
        y = calibration_pressure - y
        y/= 11.3
        KF_update(R_baro, C_baro, y)

    y = data.altd # Z measurement for simulated QC
    if y != 0.0:
        KF_update(R_sonar, C_sonar, y/1000.0)


def main():
    global x_est
    global calib_steps
    rospy.init_node('combined_filter', anonymous=True)

    rospy.Subscriber('/estimate/dnnCV', Twist, yolo_estimate_callback)
    rospy.Subscriber('/estimate/tcv', Twist, tcv_estimate_callback)
    rospy.Subscriber('/mock_gps', Twist, gps_callback)
    rospy.Subscriber('/ardrone/navdata', Navdata, navdata_callback)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)

    rospy.loginfo("Starting combined filter for estimate")


    rate = rospy.Rate(30) # Hz
    while not rospy.is_shutdown():
        if not cfg.do_calibration_before_start or calib_steps >= cfg.num_calib_steps:
            x_est = [round(i,5) for i in x_est]
            msg = hlp.to_Twist(x_est)
            filtered_estimate_pub.publish(msg)
            rospy.loginfo(v_est)
        else:
            print('calibrating')
        rate.sleep()


if __name__ == '__main__':
    main()
