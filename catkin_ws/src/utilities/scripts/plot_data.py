#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # For 3D plot
import matplotlib.gridspec as gridspec # For custom subplot grid
import numpy as np
import time
import math
import sys
import os
from scipy.signal import savgol_filter


class Data():
    def __init__(self):
        self.time = []
        self.ground_truth = []
        self.filtered_estimate = []
        self.dnnCV = []
        self.tcv = []
        self.barometer = []
        self.imu = []

        self.filtered_estimate_error = []
        self.dnnCV_error = []
        self.tcv_error = []
        self.barometer_error = []
        self.imu_error = []

    def load_data(self, filename):
        folder = './catkin_ws/src/utilities/data_storage/'
        data = np.load(folder + filename, allow_pickle=True)

        n = len(data)


        self.time = np.zeros(n)
        self.ground_truth = np.zeros((n,6))
        self.set_point = np.zeros((n,6))
        self.filtered_estimate = np.zeros((n,6))
        self.tcv = np.zeros((n,6))
        self.dnnCV = np.zeros((n,6))
        self.gps = np.zeros((n,3))
        self.barometer = np.zeros(n)
        self.imu = np.zeros((n,6))

        self.set_point_error = np.zeros((n,6))
        self.filtered_estimate_error = np.zeros((n,6))
        self.tcv_error = np.zeros((n,6))
        self.dnnCV_error = np.zeros((n,6))
        self.gps_error = np.zeros((n,3))
        self.barometer_error = np.zeros((n,6))
        self.imu_error = np.zeros((n,6))

        t_i = 0
        gt_i = 1
        set_point_i = gt_i + 6
        filtered_estimate_i = set_point_i + 6
        tcv_i = filtered_estimate_i + 6
        dnnCV_i = tcv_i + 6
        gps_i = dnnCV_i + 6
        barometer_i = gps_i + 6
        imu_i = barometer_i + 6

        set_point_error_i = imu_i + 6
        filtered_estimate_error_i = set_point_error_i + 6
        tcv_error_i = filtered_estimate_error_i + 6
        dnnCV_error_i = tcv_error_i + 6
        gps_error_i = dnnCV_error_i + 6
        barometer_error_i = gps_error_i + 6
        imu_error_i = barometer_error_i + 6

        for i, data_point in enumerate(data):
            self.time[i] = data_point[t_i]
            self.ground_truth[i,:] = data_point[gt_i:gt_i+6]
            self.set_point[i,:] = data_point[set_point_i:set_point_i+6]
            self.filtered_estimate[i,:] = data_point[filtered_estimate_i:filtered_estimate_i+6]
            self.tcv[i,:] = data_point[tcv_i:tcv_i + 6]
            self.dnnCV[i,:] = data_point[dnnCV_i:dnnCV_i+6]
            self.gps[i,:] = data_point[gps_i:gps_i + 6][0:3]
            self.barometer[i] = data_point[barometer_i:barometer_i+6][2]
            self.imu[i,:] = data_point[imu_i:imu_i+6]

            self.set_point_error[i,:] = data_point[set_point_error_i:set_point_error_i+6]
            self.filtered_estimate_error[i,:] = data_point[filtered_estimate_error_i:filtered_estimate_error_i+6]
            self.tcv_error[i,:] = data_point[tcv_error_i:tcv_error_i+6]
            self.dnnCV_error[i,:] = data_point[dnnCV_error_i:dnnCV_error_i + 6]
            self.gps_error[i,:] = data_point[gps_error_i:gps_error_i+6][0:3]
            self.barometer_error[i] = data_point[barometer_error_i:barometer_error_i+6][2]
            self.imu_error[i,:] = data_point[imu_error_i:imu_error_i+6]






def plot_xy(time, gt, est, savename):
    fig = plt.figure(figsize=(10,10))
    plt.title('Estimated trajectory xy')
    e, = plt.plot(est[:,0], est[:,1])
    gt, = plt.plot(gt[:,0], gt[:,1])
    plt.legend([e, gt], ['estimate x,y', 'ground truth x, y'])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')
    fig.tight_layout()
    # fig.show()
    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass

def error_plot(time, est, savename):
    plt.rc('font', family='Serif', size=11)
    variables = ['$x$', '$y$', '$z$', '$\psi$']

    legend_values = ['$\hat{x}$', '$\hat{y}$' ,'$\hat{z}$', '$\hat{\psi}$']
    # gt_lab = ['$x^{gt}$', '$y^{gt}$' ,'$z^{gt}$', '$\psi^{gt}$']
    subtitles = variables
    fig = plt.figure(figsize=(10,10))
    for i in range(4):
        k = i + 2 if i == 3 else i
        ax = plt.subplot(2,2,i+1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[deg]' if i == 3 else '[m]')
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.legend(legend_values[i])
        plt.grid()
        plt.title(subtitles[i])
        data_line, = ax.plot(time,est[:,k], color='r')
        data_line.set_label('estimate')
        plt.legend([data_line],[legend_values[i]])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')

    fig.tight_layout()
    # fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass


def est_plot(time, gt, est, savename):
    plt.rc('font', family='Serif', size=11)
    variables = ['$x$', '$y$', '$z$', '$\psi$']

    legend_values = ['$\hat{x}$', '$\hat{y}$' ,'$\hat{z}$', '$\hat{\psi}$']
    gt_lab = ['$x^{gt}$', '$y^{gt}$' ,'$z^{gt}$', '$\psi^{gt}$']
    subtitles = variables
    fig = plt.figure(figsize=(8.4,5))
    for i in range(4):
        k = i + 2 if i == 3 else i
        ax = plt.subplot(2,2,i+1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[deg]' if i == 3 else '[m]')
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.legend(legend_values[i])
        plt.grid()
        plt.title(subtitles[i])
        data_line, = ax.plot(time,est[:,k], color='b')
        data_line.set_label('estimate')
        gt_line, = ax.plot(time,gt[:,k], color='r')
        plt.legend([data_line, gt_line],[legend_values[i],gt_lab[i]])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')

    fig.tight_layout()
    # fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass



def est_plot_setpoint(time, gt, est, setpoint,savename):
    plt.rc('font', family='Serif', size=11)
    variables = ['$x$', '$y$', '$z$', '$\psi$']

    legend_values = ['$\hat{x}$', '$\hat{y}$' ,'$\hat{z}$', '$\hat{\psi}$']
    gt_lab = ['$x^{gt}$', '$y^{gt}$' ,'$z^{gt}$', '$\psi^{gt}$']
    sp_lab = ['$x_r$', '$y_r$' ,'$z_r$', '$\psi_r$']
    subtitles = variables
    fig = plt.figure(figsize=(8.4,5))
    for i in range(4):
        k = i + 2 if i == 3 else i
        ax = plt.subplot(2,2,i+1)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('[deg]' if i == 3 else '[m]')
        ax.axhline(y=0, color='grey', linestyle='--')
        ax.legend(legend_values[i])
        plt.grid()
        plt.title(subtitles[i])
        data_line, = ax.plot(time,est[:,k], color='b')
        data_line.set_label('estimate')
        gt_line, = ax.plot(time,gt[:,k], color='r')
        sp_line, = ax.plot(time,setpoint[:,k], color='g', linestyle="--")
        plt.legend([data_line, gt_line, sp_line],[legend_values[i],gt_lab[i], sp_lab[i]])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')

    fig.tight_layout()
    fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass



def plot_xyz(time, gt, est, savename):
    fig = plt.figure(figsize=(8.4,8.4))
    plt.rc('font', family='Serif', size=11)
    ax = fig.add_subplot(111,projection='3d')
    # plt.title('Quadcopter mission trajectory')
    est, = plt.plot(est[:,0], est[:,1], est[:,2])
    gt, = plt.plot( gt[:,0], gt[:,1], gt[:,2])
    plt.legend([est, gt], ['Estimated position', 'Ground truth position'])
    # plt.legend([est], ['Estimated position'])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')
    fig.tight_layout()
    plt.waitforbuttonpress()
    try:
        plt.waitforbuttonpress()
        # plt.close()
    except Exception as e:
        pass


def one_thing(time, est, savename):
    plt.rc('font', family='Serif', size=11)
    variables = ['$e_{tcv}$', '$y$', '$z$', '$\psi$']

    legend_values = ['$e_p$']
    subtitles = variables
    fig = plt.figure(figsize=(4,3))
    ax = plt.subplot(1,1,1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[m]')
    # ax.set_ylim([-0.4,0.4])
    ax.axhline(y=0, color='grey', linestyle='--')
    # ax.legend(legend_values[0])
    plt.grid()
    plt.title(subtitles[0])
    data_line, = ax.plot(time,est, color='b')
    data_line.set_label('estimate')
    # plt.legend([data_line],[legend_values[0]])

    folder = './catkin_ws/src/utilities/data_storage/plots/'
    plt.savefig(folder+savename+'.svg')

    fig.tight_layout()
    # fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass

def rmse(gt, data=None):
    if data is None:
        data = np.zeros_like(gt)
    return np.sqrt(np.mean((gt-data)**2))

def euc_dis(gt, data):
    data = data[:,0:3]
    gt = gt[:,0:3]
    return np.linalg.norm(gt-data, axis=1)


def bf_to_wf(bf,yaw):

    bf_xy1 = np.array([bf[0], bf[1], 1])
    wf_xy1 = np.dot(r,bf_xy1)

    wf = np.array([wf_xy1[0], wf_xy1[1], bf[2], bf[3], bf[4], bf[5]])
    return wf


def body2world(pos, ang):
    new_pos = []
    for xy,yaw in zip(pos,ang):
        x,y = xy
        yaw = yaw * math.pi/180
        c, s = math.cos(yaw), math.sin(yaw)
        r = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
        bf_xy1 = np.array([x, y, 1])
        wf_xy1 = np.dot(r,bf_xy1)
        new_pos.append(wf_xy1[0:2])
    new_pos = np.array(new_pos)
    return new_pos


def plot_real(time, tcv, dnn, barom, filtered, setpoint=None):
    plt.rc('font', family='Serif', size=11)
    variables = ['$x$', '$y$', '$z$']

    tcv_lab = ['$\hat{x}_{tcv}$', '$\hat{y}_{tcv}$' ,'$\hat{z}_{tcv}$']
    dnn_lab = ['$\hat{x}_{dnnCV}$', '$\hat{y}_{dnnCV}$' ,'$\hat{z}_{dnnCV}$']
    filtered_lab = ['$\hat{x}$', '$\hat{y}$' ,'$\hat{z}$']
    barom_lab = '$\hat{z}_{barom}$'
    setpoint_lab = ['$x_r$', '$y_r$' ,'$z_r$']
    subtitles = variables
    fig = plt.figure(figsize=(8.4,8.4))
    for i in range(3):
        ax = plt.subplot(3,1,i+1)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(variables[i] + ' [m]')
        ax.axhline(y=0, color='grey', linestyle='--')
        plt.grid()
        # plt.title(subtitles[i])
        tcv_line, = ax.plot(time,tcv[:,i], color='r')
        dnn_line, = ax.plot(time,dnn[:,i], color='g')
        if setpoint is not None:
            setpoint_line = ax.plot(time, setpoint[:,i], color='gray')
            if i == 2:
                barom_line, = ax.plot(time, barom, color='orange')
                filtered_line, = ax.plot(time, filtered[:,i], color='b')
                plt.legend([tcv_line, dnn_line, filtered_line, barom_line, setpoint_line],[tcv_lab[i],dnn_lab[i], filtered_lab[i], barom_lab, setpoint_lab[i]])
            else:
                filtered_line, = ax.plot(time, filtered[:,i], color='b')
                plt.legend([tcv_line, dnn_line, filtered_line, setpoint_line],[tcv_lab[i],dnn_lab[i], filtered_lab[i], setpoint_lab[i]])
        else:
            if i == 2:
                barom_line, = ax.plot(time, barom, color='orange')
                filtered_line, = ax.plot(time, filtered[:,i], color='b')
                plt.legend([tcv_line, dnn_line, filtered_line, barom_line],[tcv_lab[i],dnn_lab[i], filtered_lab[i], barom_lab])
            else:
                filtered_line, = ax.plot(time, filtered[:,i], color='b')
                plt.legend([tcv_line, dnn_line, filtered_line],[tcv_lab[i],dnn_lab[i], filtered_lab[i]])


    fig.tight_layout()
    fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass

def remove_dups(series):
    prevprev = series[0,:]
    prev = series[1,:]
    for i in range(series.shape[0]):
        if i < 2:
            continue
        dup = (series[i,:] == prev).all() and (series[i,:] == prevprev).all()
        prevprev = np.copy(prev)
        prev = np.copy(series[i,:])
        if dup:
            series[i,:] = None
    return series

def main():
    data = Data()
    data.load_data('sim_hover.npy')
    # print(data.ground_truth)
    # data.set_point[0:55,0] = 2
    # data.set_point[0:55,1] = 2
    # data.set_point[0:55,2] = 5
    # data.set_point[0:55,5] = -90


    # print(data.dnnCV)

    # data.dnnCV = remove_dups(data.dnnCV)
    # data.tcv = remove_dups(data.tcv)

    data.ground_truth[:,0:2] = body2world(data.ground_truth[:,0:2], data.ground_truth[:,5])
    data.filtered_estimate[:,0:2] = body2world(data.filtered_estimate[:,0:2], data.filtered_estimate[:,5])


    for i in range(0,6):
        data.filtered_estimate[:,i] = savgol_filter(data.filtered_estimate[:,i], 13, 3)

    # m = 450
    # data.time = data.time[:m]
    # data.dnnCV = data.dnnCV[:m,:]
    # data.tcv = data.tcv[:m,:]
    # data.barometer = data.barometer[:m]
    # data.filtered_estimate = data.filtered_estimate[:m,:]
    # data.ground_truth = data.ground_truth[:m,:]
    # plot_real(data.time, data.tcv, data.dnnCV, data.barometer, data.filtered_estimate)
    # plot_xyz(data.time, data.ground_truth,data.filtered_estimate, "nad")

    # est_plot_setpoint(data.time, data.ground_truth, data.filtered_estimate, data.set_point,"landing_est_gt_setpoint")
    # est_plot_setpoint(data.time, data.dnnCV, data.filtered_estimate, data.tcv, "nad")
    # est_plot(data.time, data.ground_truth, data.filtered_estimate, "testplot")
    # est_plot(data.time, data.ground_truth, data.tcv, "testplot")
    one_thing(data.time,euc_dis(data.ground_truth, data.tcv), "name")
    # error_plot(data.time, data.ground_truth - data.filtered_estimate, "testplot")

    # plot_xyz(data.time, data.ground_truth, data.filtered_estimate, '3dplot')

    print('rmse filtered: ' , rmse(euc_dis(data.ground_truth, data.filtered_estimate)))
    print('rmse filtered yaw: ' , rmse(data.ground_truth[:,5], data.filtered_estimate[:,5]))
    print('rmse dnncv: ' , rmse(euc_dis(data.ground_truth, data.dnnCV)))
    print('rmse dnncv yaw: ' , rmse(data.ground_truth[:,5], data.dnnCV[:,5]))
    print('rmse tcv: ' , rmse(euc_dis(data.ground_truth[:,0:3], data.tcv)))
    print('rmse tcv yaw: ' , rmse(data.ground_truth[:,5], data.tcv[:,5]))
    print('rmse gps: ' , rmse(euc_dis(data.ground_truth[:,0:3], data.gps)))
    print('rmse barom: ' , rmse(data.ground_truth[:,2], data.barometer))
    print('rmse imu: ' , rmse(euc_dis(data.ground_truth, data.imu)))
    print('rmse imu yaw: ' , rmse(data.ground_truth[:,5], data.imu[:,5]))
    print('----')

    print('mean filtered: ' , -np.mean(data.ground_truth[:,0:3] - data.filtered_estimate[:,0:3]))
    print('mean filtered yaw: ' , -np.mean(data.ground_truth[:,5] - data.filtered_estimate[:,5]))
    print('mean dnncv: ' ,- np.mean(data.ground_truth[:,0:3] - data.dnnCV[:,0:3]))
    print('mean dnncv yaw: ' ,- np.mean(data.ground_truth[:,5] -  data.dnnCV[:,5]))
    print('mean tcv: ' , -np.mean(data.ground_truth[:,0:3] - data.tcv[:,0:3]))
    print('mean tcv yaw: ' ,-np.mean(data.ground_truth[:,5] - data.tcv[:,5]))
    print('mean gps: ' , -np.mean(data.ground_truth[:,0:3] - data.gps))
    print('mean barom: ' ,-np.mean(data.ground_truth[:,2] - data.barometer))
    print('mean imu: ' , -np.mean(data.ground_truth[:,0:3] - data.imu[:,0:3]))
    print('mean imu yaw: ' , -np.mean(data.ground_truth[:,5] - data.imu[:,5]))
    print('----')
    print('std filtered: ' , np.std(data.ground_truth[:,0:3] - data.filtered_estimate[:,0:3]))
    print('std filtered yaw: ' , np.std(data.ground_truth[:,5] - data.filtered_estimate[:,5]))
    print('std dnncv: ' , np.std(data.ground_truth[:,0:3] - data.dnnCV[:,0:3]))
    print('std dnncv yaw: ' , np.std(data.ground_truth[:,5] -  data.dnnCV[:,5]))
    print('std tcv: ' , np.std(data.ground_truth[:,0:3] - data.tcv[:,0:3]))
    print('std tcv yaw: ' , np.std(data.ground_truth[:,5] - data.tcv[:,5]))
    print('std gps: ' , np.std(data.ground_truth[:,0:3] - data.gps[:,0:3]))
    print('std barom: ' , np.std(data.ground_truth[:,2] - data.barometer))
    print('std imu: ' , np.std(data.ground_truth[:,0:3] -  data.imu[:,0:3]))
    print('std imu yaw: ' , np.std(data.ground_truth[:,5] - data.imu[:,5]))




if __name__ == '__main__':
    main()
