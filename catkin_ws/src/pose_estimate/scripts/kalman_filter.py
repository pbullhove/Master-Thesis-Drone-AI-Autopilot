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
- Barometer: drone altitude
- Mock gps: in the simulated environmet, not available on real ardrone.

Tune Kalman Filter by changing R-values for the desired sensors, where R specifies measurement noise,
and by changing Q_imu which specifies how uncertain IMU predictions are.

Subscribes to:
    /estimate/dnnCV: Twist - Pose estimates from the dnn CV method
    /estimate/tcv: Twist - Pose estimates from the tcv CV method.
    /mock_gps: Twist - Pose estimates from the simulator mock gps.
    /ardrone/navdata: Odometry - Odometry data from ardrone.
    /ardrone/takeoff: Empty - To reset x_est and v_est on takeoff.

Publishes to:
    /filtered_estimate: Twist - the current estimated quadcopter pose
    /filtered_estimate_vel: Twist - the current estimated quadcopter velocity.
"""
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from ardrone_autonomy.msg import Navdata
from sensor_msgs.msg import Imu, Range
from std_msgs.msg import Empty
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

C_dnnCV = np.eye(6)
C_tcv = np.eye(6)
C_gps = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
C_baro = np.array([0,0,1,0,0,0]).reshape((1,6))

R_dnnCV = (0.4**2)*np.eye(6) # 10hz
R_tcv = (0.1**2)*np.diag([1,1,1,1,1,1000]) # 10hz
R_gps = (0.1**2)*np.eye(3) # 2 hz
R_baro = (0.7**2)*np.eye(1) # 200hz

Q_imu = 0.01*np.diag([1, 1, 1, 0, 0, 1])
Q_imu_vel = 0.01*np.diag([1, 1, 1])


calibration_vel = np.array([0.0, 0.0, 0.0])
calibration_acc = np.array([0.0, 0.0, 0.0])
calibration_pressure = 0.0
calib_pitch = 0.0
calib_roll = 0.0
calib_steps = 0
low_acc_limit = 0.1
high_acc_limit = 5.0
ONE_g = 9.8067
vel_decay = 0.0
prev_imu_yaw = None
prev_navdata_timestamp = None

y_prev_range = None
t_prev_range = None
y_prev_tcv = None
t_prev_tcv = None
y_prev_gps = None
t_prev_gps = None
y_prev_dnnCV = None
t_prev_dnnCV = None

imu_integrated = np.zeros(6)
barometric_altitude = np.zeros(6)

def kalman_gain(P,C,R):
    """ Computes the Kalman Gain which specifies how much to update x_est given a new measurement. """
    PCT = np.dot(P, C.T)
    IS = R + np.dot(C, PCT)
    IS_inv = np.linalg.inv(IS)
    K = np.dot(PCT,IS_inv)
    return K

def P_post(P, C, K):
    """ Updating P after updateing x_est on new data"""
    P = np.dot((np.eye(np.shape(P)[0]) - np.dot(K,C)), P)
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
    """ If measurement is more than 5m away from current estimate and is more than 5m of start: discard measurement"""
    try:
        if ((np.absolute(y - np.dot(C,x_est)))[0:3] > np.array([5,5,5])).any() and (np.absolute(y)[0:3] > np.array([5,5,5])).any():
            print('discard')
            return
    except IndexError as e: # range updates
        if (np.absolute(y - np.dot(C,x_est)) > 5) and (np.absolute(y) > 5):
            return

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

def KF_vel_update(R, C, y, y_prev, t_prev):
    global v_est
    global P_v
    C = C[:,0:3]
    R = R[:,0:3]
    now = datetime.now()
    delta_t = (now - t_prev).total_seconds()
    y_vel = (y - y_prev)/delta_t
    y_vel = np.array([min(cfg.vel_estimate_limit, i) for i in y_vel])
    y_vel = np.array([max(-cfg.vel_estimate_limit, i) for i in y_vel])
    K = kalman_gain(P_v, C, R)
    vel_innov = y_vel - np.dot(C, v_est)
    vel_innov = np.array([min(cfg.vel_innov_limit, i) for i in vel_innov])
    vel_innov = np.array([max(-cfg.vel_innov_limit, i) for i in vel_innov])
    v_est = v_est + np.dot(K, vel_innov)
    P_v = P_post(P_v,C,K)
    return y, now


def dnnCV_estimate_callback(data):
    """ Filters pose estimates from dnnCV cv algorithm. Estimates pos in xyz and yaw. Only use this if more than 0.7m above platform, as camera view too close for correct estimates. """
    global t_prev_dnnCV
    global y_prev_dnnCV
    dnnCV_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.7 or dnnCV_estimate[2] > 0.7:
        if dnnCV_estimate[5] == 0.0: #if no estimate for yaw
            C = C_dnnCV[0:3,:]
            y = dnnCV_estimate[0:3]
            R = R_dnnCV[0:3,0:3]
        else:
            C = C_dnnCV
            R = R_dnnCV
            y = dnnCV_estimate
        KF_update(R,C,y)
        try:
            if (datetime.now() - t_prev_dnnCV).total_seconds() < 1:
                y_prev_dnnCV, t_prev_dnnCV = KF_vel_update(R_dnnCV[0:3,0:3]*4, C[0:3,0:3], y[0:3], y_prev_dnnCV, t_prev_dnnCV)
            else:
                t_prev_dnnCV = datetime.now()
                y_prev_dnnCV = y[0:3]
        except TypeError  as e:
            t_prev_dnnCV = datetime.now()
            y_prev_dnnCV = y[0:3]

def tcv_estimate_callback(data):
    global t_prev_tcv
    global y_prev_tcv
    """ Filters pose estimates from tcv cv algorithm. Estimates pos in xyz and yaw. Only use this if mmore than 0.4m above platform, as camera view too close for correct estimates. """
    tcv_estimate = hlp.twist_to_array(data)
    if x_est[2] > 0.4 or tcv_estimate[2] > 0.4:
        if tcv_estimate[5] == 0.0 or tcv_estimate[5] == -0.0: #if no estimate for yaw
            C = C_dnnCV[0:3,:]
            y = tcv_estimate[0:3]
            R = R_tcv[0:3,0:3]
        else:
            C = C_tcv
            R = R_tcv
            y = tcv_estimate
        KF_update(R,C,y)
        try:
            if (datetime.now() - t_prev_tcv).total_seconds() < 1:
                y_prev_tcv, t_prev_tcv = KF_vel_update(R_tcv[0:3,0:3]*10, C[0:3,0:3], y[0:3], y_prev_tcv, t_prev_tcv)
            else:
                t_prev_tcv = datetime.now()
                y_prev_tcv = y[0:3]
        except TypeError  as e:
            t_prev_tcv = datetime.now()
            y_prev_tcv = y[0:3]



def gps_callback(data):
    """ Filters gps data which is measurement of position in xyz. """
    global t_prev_gps
    global y_prev_gps
    gps_measurement = hlp.twist_to_array(data)
    y = gps_measurement[0:3]
    KF_update(R_gps, C_gps, y)
    try:
        if (datetime.now() - t_prev_gps).total_seconds() < 1:
            y_prev_gps, t_prev_gps = KF_vel_update(R_gps*4, C_gps, y, y_prev_gps, t_prev_gps)
        else:
            t_prev_gps = datetime.now()
            y_prev_gps = y
    except TypeError  as e:
        t_prev_gps = datetime.now()
        y_prev_gps = y

def takeoff_callback(data):
    """ Resets estimate on takeoff """
    global x_est
    global v_est
    global P
    global P_v
    x_est[0:3] = np.array([0.0,0.0,0.0])
    P[0:3,0:3] = 0.1*np.eye(3)
    v_est = np.array([0.0,0.0,0.0])
    P_v = 0.1*np.eye(3)

def navdata_callback(data):
    """
    Filters estimates from the IMU data and predicts quadcopter pose.
    Updates estimate with barometric pressure data.

    Performs calibration before start by setting calibration_vel, calibration_acc to average of standstill values for about 5 seconds.
    Does not publish any estimates before after calibration. Toggle calibration in config.

    input:
        data: Odometry from ardrone 200hz
    output:
        x_est, P, v_est, P_v
    """
    global x_est
    global v_est
    global P
    global P_v
    global prev_imu_yaw
    global calib_steps
    global prev_navdata_timestamp
    global calibration_vel
    global calibration_acc
    global calibration_pressure
    global calib_roll
    global calib_pitch
    global t_prev_range
    global y_prev_range

    """ Calibration before start. """
    if cfg.do_calibration_before_start and calib_steps < cfg.num_calib_steps:
        calibration_vel += np.array([data.vx, data.vy, data.vz])/float(cfg.num_calib_steps)
        calibration_acc += np.array([data.ax, data.ay, data.az - 1])*ONE_g/float(cfg.num_calib_steps)
        calibration_pressure += data.pressure/float(cfg.num_calib_steps)
        calib_roll += data.rotX/float(cfg.num_calib_steps)
        calib_pitch += data.rotY/float(cfg.num_calib_steps)
        calib_steps += 1
        return

    """ Reading yaw and time data. """
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

    """ Reading IMU accelerations. """
    acc = np.array([data.ax*ONE_g, data.ay*ONE_g, data.az*ONE_g]) - calibration_acc
    p = hlp.deg2rad(data.rotY - calib_pitch)
    r = hlp.deg2rad(data.rotX - calib_roll)
    gravity_vec = ONE_g*np.array([-math.sin(p),math.cos(p)*math.sin(r), math.cos(p)*math.cos(r)])
    acc -= gravity_vec
    low_values_filter_acc = np.logical_and(np.less(acc, low_acc_limit), np.greater(acc, -low_acc_limit))
    high_values_filter_acc = np.logical_and(np.less(acc, high_acc_limit), np.greater(acc, -high_acc_limit))
    acc[low_values_filter_acc] = 0.0
    acc[low_values_filter_acc] = 0.0

    """ Integrating acc to find v_est. """
    v_est *= 1-vel_decay
    v_est += delta_t*acc
    v_est = np.array([max(-cfg.vel_estimate_limit,i) for i in v_est])
    v_est = np.array([min(cfg.vel_estimate_limit,i) for i in v_est])
    delta_pos = delta_t*v_est
    delta_pos = np.zeros(3)

    """ Integrating IMU data for plotting and tuning"""
    imu_integrated[0:3] += delta_pos
    imu_integrated[3:] = data.rotX, data.rotY, data.rotZ
    imu_integrated[0:3] = R.from_euler('z', -np.radians(delta_yaw)).apply(imu_integrated[0:3])


    """ Predicting x_est based on v_est and delta_yaw."""
    delta_x = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0, 0, delta_yaw])
    x_est = x_est + delta_x
    r = R.from_euler('z', -np.radians(delta_yaw))
    x_est[0:3] = r.apply(x_est[0:3])
    P = P_apri(P, Q_imu)
    P_v = P_apri(P_v, Q_imu_vel)
    x_est[3] = data.rotX
    x_est[4] = data.rotY
    x_est[5] = hlp.angleFromTo(x_est[5],-180,180)

    """ Update x_est and v_est based or barometric data. """
    y = data.pressure # Z measurement for real QC
    if cfg.is_simulator: # simulated qc does not have barometer. Use simulated height to generate sonar data.
        y = - data.altd * 11.3 / 1000.0
    if y > 0.0:
        y = calibration_pressure - y
        y /= 11.3 # pressure to height constant
        KF_update(R_baro, C_baro, y)
        barometric_altitude = y
        try:
            if (datetime.now() - t_prev_range).total_seconds() < 1:
                if np.absolute(y - y_prev_range) > 0.01:
                    y_prev_range, t_prev_range = KF_vel_update(R_baro*10, C_baro, y, y_prev_range, t_prev_range)
            else:
                t_prev_range = datetime.now()
                y_prev_range = y
        except TypeError  as e:
            t_prev_range = datetime.now()
            y_prev_range = y


def main():
    rospy.init_node('combined_filter', anonymous=True)

    rospy.Subscriber('/estimate/dnnCV', Twist, dnnCV_estimate_callback)
    rospy.Subscriber('/estimate/tcv', Twist, tcv_estimate_callback)
    rospy.Subscriber('/mock_gps', Twist, gps_callback)
    rospy.Subscriber('/ardrone/navdata', Navdata, navdata_callback)
    rospy.Subscriber('/ardrone/takeoff', Empty, takeoff_callback)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)
    filtered_vel_pub = rospy.Publisher('/filtered_estimate_vel', Twist, queue_size=10)
    pub_imu_integrated = rospy.Publisher('/imu_integrated', Twist, queue_size=10)
    pub_barom = rospy.Publisher('/barometric_altitude', Twist, queue_size=10)

    rospy.loginfo("Starting combined filter for estimate")


    rate = rospy.Rate(30) # Hz
    while not rospy.is_shutdown():
        if not cfg.do_calibration_before_start or calib_steps >= cfg.num_calib_steps:
            x = [round(i,5) for i in x_est]
            msg = hlp.to_Twist(x)
            v = [round(i,5) for i in v_est] + [0,0,0]
            vel_msg = hlp.to_Twist(v)
            filtered_estimate_pub.publish(msg)
            filtered_vel_pub.publish(vel_msg)
            pub_imu_integrated.publish(hlp.to_Twist(imu_integrated))
            pub_barom.publish(hlp.to_Twist(barometric_altitude))
            # rospy.loginfo(v_est)
        else:
            print('calibrating')
        rate.sleep()


if __name__ == '__main__':
    main()
