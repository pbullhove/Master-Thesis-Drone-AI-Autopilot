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

R_dnnCV = (0.02**2)*np.eye(6) # 10hz
R_tcv = (0.01**2)*np.diag([1,1,1,1,1,1000]) # 10hz
R_gps = (0.1**2)*np.eye(3) # 2 hz
R_baro = (0.3**2)*np.eye(1) # 200hz

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
median_filter_size = 1
average_filter_size = 1
barom_median_filter_size = 5
barom_average_filter_size = 5
estimate_history_size = median_filter_size + average_filter_size - 1
estimate_history_dnn = np.zeros((estimate_history_size,3))
estimate_history_tcv = np.zeros((estimate_history_size,3))
estimate_history_dnn_yaw = np.zeros(estimate_history_size)
estimate_history_tcv_yaw = np.zeros(estimate_history_size)
barom_estimate_history_size = barom_median_filter_size + barom_average_filter_size - 1
estimate_history_barom = np.zeros(barom_estimate_history_size)

def filter_estimate(estimate, estimate_history, median_filter_size, average_filter_size):
    """
        Filters the estimate with a sliding window median and average filter.
    """

    estimate_history = np.concatenate((estimate_history[1:], [estimate]))

    strides = np.array(
        [estimate_history[i:median_filter_size+i] for i in range(average_filter_size)]
    )

    median_filtered = np.median(strides, axis = 1)
    average_filtered = np.average(median_filtered[-average_filter_size:], axis=0)

    return average_filtered, estimate_history


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
    # """ If measurement is more than 5m away from current estimate and is more than 5m of start: discard measurement"""
    # try:
    #     if ((np.absolute(y - np.dot(C,x_est)))[0:3] > np.array([5,5,5])).any() and (np.absolute(y)[0:3] > np.array([5,5,5])).any():
    #         print('discard')
    #         return
    # except IndexError as e: # range updates
    #     if (np.absolute(y - np.dot(C,x_est)) > 5) and (np.absolute(y) > 5):
    #         return

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
    global estimate_history_dnn
    global estimate_history_dnn_yaw
    dnnCV_estimate = hlp.twist_to_array(data)
    dnnCV_estimate[0:3], estimate_history_dnn = filter_estimate(dnnCV_estimate[0:3], estimate_history_dnn, median_filter_size, average_filter_size)
    if dnnCV_estimate[5] != 0.0:
        dnnCV_estimate[5], estimate_history_dnn_yaw = filter_estimate(dnnCV_estimate[5], estimate_history_dnn_yaw, median_filter_size, average_filter_size)
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
    global estimate_history_tcv
    global estimate_history_tcv_yaw
    """ Filters pose estimates from tcv cv algorithm. Estimates pos in xyz and yaw. Only use this if mmore than 0.4m above platform, as camera view too close for correct estimates. """
    tcv_estimate = hlp.twist_to_array(data)
    tcv_estimate[0:3], estimate_history_tcv = filter_estimate(tcv_estimate[0:3], estimate_history_tcv, median_filter_size, average_filter_size)
    if tcv_estimate[5] != 0.0:
        tcv_estimate[5], estimate_history_tcv_yaw = filter_estimate(tcv_estimate[5], estimate_history_tcv_yaw, median_filter_size, average_filter_size)

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


gps_data = np.zeros(3)
def gps_callback(data):
    """ Filters gps data which is measurement of position in xyz. """
    global t_prev_gps
    global y_prev_gps
    global gps_data
    gps_measurement = hlp.twist_to_array(data)
    y = gps_measurement[0:3]
    gps_data = y
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

def reset_imu_values(data):
    global imu_integrated
    imu_integrated = x_est

# def update_yaw(data):
#     global x_est
#     innov = hlp.angleFromTo(data.angular.z - x_est[5], -180,180)
#     x_est[5] = x_est[5] + 0.05*(innov)
#     x_est[5] = hlp.angleFromTo(x_est[5], -180,180)

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
    global estimate_history_barom

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
        delta_yaw = (data.rotZ - 90) - prev_imu_yaw + 0.03*(data.rotZ - 90 - x_est[5])
        delta_yaw = hlp.angleFromTo(delta_yaw, -180,180)
        now = datetime.now()
        delta_t = (now - prev_navdata_timestamp).total_seconds()
        prev_navdata_timestamp = now
    except TypeError as e: #first iteration
        delta_yaw = 0
        delta_t = 0
        prev_navdata_timestamp = datetime.now()
    finally:
        prev_imu_yaw = (data.rotZ - 90)

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
    # v_est = np.array([data.vx, data.vy, data.vz])/1000.0
    # v_est = np.array([max(-cfg.vel_estimate_limit,i) for i in v_est])
    # v_est = np.array([min(cfg.vel_estimate_limit,i) for i in v_est])
    delta_pos = delta_t*v_est
    # delta_pos = np.zeros(3)

    """ Integrating IMU data for plotting and tuning"""
    imu_integrated[0:3] += delta_t*v_est
    imu_integrated[3:] = data.rotX, data.rotY, imu_integrated[5]+delta_yaw
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
    if cfg.is_simulator:
        y = data.altd / 1000.0
    elif data.pressure > 0:
        y = (calibration_pressure - data.pressure)/11.3
    if y > 0.0 and data.altd < 2500.0:
        y, estimate_history_barom = filter_estimate(y, estimate_history_barom, barom_median_filter_size, barom_average_filter_size)
        KF_update(R_baro, C_baro, y)
        barometric_altitude[2] = y
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
    rospy.Subscriber('/start_data_collection', Empty, reset_imu_values)
    # rospy.Subscriber('/drone_ground_truth', Twist, update_yaw)

    filtered_estimate_pub = rospy.Publisher('/filtered_estimate', Twist, queue_size=10)
    filtered_vel_pub = rospy.Publisher('/filtered_estimate_vel', Twist, queue_size=10)
    pub_imu_integrated = rospy.Publisher('/estimate/imu', Twist, queue_size=10)
    pub_barom = rospy.Publisher('/estimate/barometer', Twist, queue_size=10)
    pub_gps = rospy.Publisher('/estimate/gps', Twist, queue_size=10)


    rospy.loginfo("Starting combined filter for estimate")


    rate = rospy.Rate(30) # Hz
    while not rospy.is_shutdown():
        if not cfg.do_calibration_before_start or calib_steps >= cfg.num_calib_steps:
            x = [round(i,5) for i in x_est]
            # x[5] += 1.0 #some offset in estimates
            msg = hlp.to_Twist(x)
            v = [round(i,5) for i in v_est] + [0,0,0]
            vel_msg = hlp.to_Twist(v)
            filtered_estimate_pub.publish(msg)
            filtered_vel_pub.publish(vel_msg)
            pub_imu_integrated.publish(hlp.to_Twist(imu_integrated))
            pub_barom.publish(hlp.to_Twist(barometric_altitude))
            if gps_data is not None:
                gps = [round(i,5) for i in gps_data] + [0,0,0]
                pub_gps.publish(hlp.to_Twist(gps))
        else:
            print('calibrating')
        rate.sleep()


if __name__ == '__main__':
    main()
