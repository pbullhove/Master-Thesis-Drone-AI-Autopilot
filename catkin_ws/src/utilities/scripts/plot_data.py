#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # For 3D plot
import matplotlib.gridspec as gridspec # For custom subplot grid
import numpy as np
import time
import math
import sys
import os



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
    fig = plt.figure(figsize=(12,8))
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
    fig = plt.figure(figsize=(12,8))
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
    fig = plt.figure(figsize=(12,8))
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
    # fig.show()

    try:
        plt.waitforbuttonpress(0)
        plt.close()
    except Exception as e:
        pass



def plot_xyz(time, gt, est, savename):
    fig = plt.figure(figsize=(10,10))
    plt.rc('font', family='Serif', size=11)
    ax = fig.add_subplot(111,projection='3d')
    # plt.title('Quadcopter mission trajectory')
    est, = plt.plot(est[:,0], est[:,1], est[:,2])
    gt, = plt.plot( gt[:,0], gt[:,1], gt[:,2])
    plt.legend([est, gt], ['Estimated position', 'Ground truth position'])

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
    variables = ['$e_{estimate}$', '$y$', '$z$', '$\psi$']

    legend_values = ['$e_p$']
    subtitles = variables
    fig = plt.figure(figsize=(12,8))
    ax = plt.subplot(1,1,1)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[m]')
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



def main():
    data = Data()
    data.load_data('unnamed.npy')
    # data.set_point[0:55,0] = -2
    # data.set_point[0:55,1] = 2
    # data.set_point[0:55,2] = 5
    # data.set_point[0:55,5] = -90
    # est_plot_setpoint(data.time, data.ground_truth, data.filtered_estimate, data.set_point,"landing_est_gt_setpoint")
    data.ground_truth[:,0:2] = body2world(data.ground_truth[:,0:2], data.ground_truth[:,5])
    data.filtered_estimate[:,0:2] = body2world(data.filtered_estimate[:,0:2], data.filtered_estimate[:,5])
    est_plot(data.time, data.ground_truth, data.filtered_estimate, "testplot")
    # error_plot(data.time, data.ground_truth - data.imu, "testplot")
    # one_thing(data.time,euc_dis(data.filtered_estimate, data.ground_truth), "name")

    plot_xyz(data.time, data.ground_truth, data.filtered_estimate, '3dplot')

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
    print('mean filtered error: ' , np.mean(euc_dis(data.ground_truth, data.filtered_estimate)))
    print('mean filtered yaw error: ' , np.mean(data.ground_truth[:,5] - data.filtered_estimate[:,5]))
    print('mean dnncv error: ' , np.mean(euc_dis(data.ground_truth, data.dnnCV)))
    print('mean dnncv yaw error: ' , np.mean(data.ground_truth[:,5] -  data.dnnCV[:,5]))
    print('mean tcv error: ' , np.mean(euc_dis(data.ground_truth, data.tcv)))
    print('mean tcv yaw error: ' , np.mean(data.ground_truth[:,5] - data.tcv[:,5]))
    print('mean gps error: ' , np.mean(euc_dis(data.ground_truth[:,0:3], data.gps)))
    print('mean barom error: ' , np.mean(data.ground_truth[:,2] - data.barometer))
    print('mean imu error: ' , np.mean(euc_dis(data.ground_truth, data.imu)))
    print('mean imu yaw error: ' , np.mean(data.ground_truth[:,5] - data.imu[:,5]))
    print('----')
    print('std filtered error: ' , np.std(euc_dis(data.ground_truth, data.filtered_estimate)))
    print('std filtered yaw error: ' , np.std(data.ground_truth[:,5]- data.filtered_estimate[:,5]))
    print('std dnncv error: ' , np.std(euc_dis(data.ground_truth, data.dnnCV)))
    print('std dnncv yaw error: ' , np.std(data.ground_truth[:,5] -  data.dnnCV[:,5]))
    print('std tcv error: ' , np.std(euc_dis(data.ground_truth[:,0:3], data.tcv)))
    print('std tcv yaw error: ' , np.std(data.ground_truth[:,5] - data.tcv[:,5]))
    print('std gps error: ' , np.std(euc_dis(data.ground_truth[:,0:3], data.gps)))
    print('std barom error: ' , np.std(data.ground_truth[:,2] - data.barometer))
    print('std imu error: ' , np.std(euc_dis(data.ground_truth, data.imu)))
    print('std imu yaw error: ' , np.std(data.ground_truth[:,5] - data.imu[:,5]))




    # est_plot(data.time, data.ground_truth, data.imu, "testplot")

    # for file in os.listdir("./catkin_ws/src/uav_vision/data_storage"):
    #     try:
    #         loadname = file
    #         savename = loadname.split('.')[0]
    #         time, gt, yolo, tcv, yolo_error, filtered = load_data(loadname)
    #     except Exception as e:
    #         continue
    #
    #     if savename.split('_')[-1] == 'comb':
    #         est_plot_comb(time, gt, yolo, tcv, savename + "_var_plot")
    #         est_plot(time, gt, filtered, savename + "_filtered_var_plot")
    #         est_plot_comb_filtered(time, gt, yolo, tcv, filtered, savename + "_comb_plot")
    #     elif savename.split('_')[-1] == 'yolo':
    #         est_plot(time, gt, filtered, savename +"_filtered_plot")
    #     elif savename.split('_')[-1] == 'tcv':
    #         est_plot(time, gt, filtered, savename + "_var_plot")
    #     else:
    #         est_plot_comb(time, gt, yolo, tcv, savename + "_var_plot")
    #         est_plot(time, gt, filtered, savename + "_filtered_var_plot")
    #     if savename.split('_')[0] in ['landing','land']:
    #         plot_xy(time, gt, filtered, savename+"_xy_plot")
    #         plot_xyz(time, gt, filtered, savename+"_xyz_plot")


if __name__ == '__main__':
    main()
