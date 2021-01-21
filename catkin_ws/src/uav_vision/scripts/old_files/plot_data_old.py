#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # For 3D plot
import matplotlib.gridspec as gridspec # For custom subplot grid
import numpy as np
import time
import sys

# Variables
V_X = 0
V_Y = 1
V_Z = 2
V_YAW = 5

# List if indices
## 0    time_stamps

## 1    ground_truth

## 7    est_ellipse
## 13	est_arrow
## 19	est_corners
## 25	est_dead_reckoning

## 31	est_error_ellipse
## 37	est_error_arrow
## 43	est_error_corners
## 49	est_error_dead_reckoning

## 55   filtered_estimate

def plot_data(stored_array, methods_to_plot, variables_to_plot, plot_error=False, plot_z_to_the_right=False, z_right_color='g'):
    t_id = 0            # Time index
    g_id = 1            # Ground truth index
    e_id = g_id + 6     # Ellipse index
    a_id = e_id + 6     # Arrow index
    c_id = a_id + 6     # Corner index
    d_id = c_id + 6     # Dead reckoning index

    error_e_id = d_id + 6     # Ellipse error index
    error_a_id = error_e_id + 6     # Arrow error index
    error_c_id = error_a_id + 6     # Corner error index
    error_d_id = error_c_id + 6     # Dead reckoning error index

    time_stamps = stored_array[:, t_id]

    titles_variables = [
        "x-Position", "y-Position", "z-Position", "None", "None", "yaw-Rotation"
    ]
    titles_error_variables = [
        "x-Position Error", "y-Position Error", "z-Position Error",
        "none", "none", "yaw-Rotation Error"
    ]

    lables_variables = [
        "x-Position [m]", "y-Position [m]", "z-Position [m]", "none", "none", "yaw-Rotation [deg]",
    ]
    lables_error_variables = [
        "x-Position Error [m]", "y-Position Error [m]", "z-Position Error [m]", "none", "none", "yaw-Rotation Error [deg]",
    ]
    titles_methods = [
        "Ground truth",
        "Ellipse",
        "Arrow",
        "Corners",
        "Dead reckogning",
        "Ellipse error",
        "Arrow error",
        "Corners error",
        "Dead reckogning error",
    ]
    indices_methods = [g_id, e_id, a_id, c_id, d_id,
        error_e_id, error_a_id, error_c_id, error_d_id
    ]
    colors_methods = [
        "g",        # green:    "Ground truth"
        "b",        # blue:     "Ellipse"
        "r",        # red:      "Arrow"
        "orange",   # orange:   "Corners"
        "k",        # black:    "Dead reckogning"
        "b",        # blue:     "Ellipse error"
        "r",        # red:      "Arrow error"
        "orange",   # orange:   "Corners error"
        "k"         # black:    "Dead reckogning error"
    ]
    y_ticks_error_pos = np.arange(-0.10, 0.11, 0.025)
    y_ticks_error_rot = np.arange(-10, 11, 2)
    y_ticks_error = np.array([
        [y_ticks_error_pos]*3, [y_ticks_error_rot]*3
    ])

    for variable in variables_to_plot:

        if plot_error:
            title = titles_error_variables[variable]
            y_label = lables_error_variables[variable]
        else:
            title = titles_variables[variable]
            y_label = lables_variables[variable]

        fig, ax = plt.subplots(figsize=(10,8))
        # fig, ax = plt.subplots(figsize=(20,15))

        if plot_error:
            ax.axhline(y=0, color='grey', linestyle='--') # Plot the zero-line


        for method in methods_to_plot:
            legend_text = titles_methods[method]
            line_color = colors_methods[method]
            index = indices_methods[method]

            data = stored_array[:, index:index+6][:,variable]
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan
      
            line, = ax.plot(time_stamps_local, data)
            line.set_color(line_color)
            line.set_label(legend_text)

            ax.set_title(title)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel(y_label)
            ax.legend(loc='upper left', facecolor='white', framealpha=1)

            # if plot_error:
            #     ax.set_yticks(y_ticks_error[variable])

            # ax.xaxis.grid()
            # ax.yaxis.grid()
            # ax.grid()
        
        if plot_z_to_the_right:
            # Plot the z ground truth
            gt_method = 0
            z_variable = 2

            index = indices_methods[gt_method]
            data = stored_array[:, index:index+6][:, z_variable]
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            ax2 = ax.twinx()
            line, = ax2.plot(time_stamps_local, data)
            line.set_color(z_right_color)
            line.set_label("Ground truth z-Position")

            ax2.legend(loc='upper right', facecolor='white', framealpha=1)

            ax2.set_ylabel('z-Position [m]', color=z_right_color)
            ax2.set_yticks(np.arange(7))
            ax2.tick_params(axis='y', labelcolor=z_right_color)
            ax2.grid(None)

        plt.xlim(time_stamps[0], time_stamps[-1])
        plt.grid()

        fig.tight_layout()
        
        folder = './plots/'
        plt.savefig(folder+title+'.svg')


        fig.draw
        plt.waitforbuttonpress(0)
        plt.close()

        # plt.show()

def plot_data_manually(stored_array):
    index_values = [1, 7, 13, 19]
    color_values = ['green', 'blue', 'red', 'orange']
    legend_values = ['ground truth', 'ellipse', 'arrow', 'corners']
    legend_lines = []

    index_errors = [31, 37, 43]
    color_errors = ['blue', 'red', 'orange']

    time_stamps = stored_array[:, 0]
    ground_truth = stored_array[:, 1:1+6]

    fig = plt.figure(figsize=(10,12))

    for i in range(1,9):
        ax = plt.subplot(4,2,i)
        plt.grid()
        plt.xlim(time_stamps[0], time_stamps[-1])

        # Plot the ground truth value for z
        if i != 5:
            right_ax = ax.twinx()

            legend_text = 'ground truth z-position'
            z_right_color = 'lightgrey'

            data = ground_truth[:,V_Z]
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = right_ax.plot(time_stamps_local, data)
            line.set_color(z_right_color)

            right_ax.set_ylabel('z-position [m]', color='grey')
            right_ax.tick_params(axis='y', labelcolor='grey')
            # right_ax.set_ylim(-0.07, 5.84)

        if i == 1:
            variable = V_X
            y_label = "x-position [m]"
            y_tics = np.linspace(-0.25, 0.05, num=7)
        elif i == 2:
            variable = V_X
            y_label = "x-position error [m]"
            y_tics = np.linspace(-0.10, 0.10, num=5)
        elif i == 3:
            variable = V_Y
            y_label = "y-position [m]"
            y_tics = np.linspace(-0.15, 0.15, num=7)
        elif i == 4:
            variable = V_Y
            y_label = "y-position error[m]"
            y_tics = np.linspace(-0.10, 0.10, num=5)
        elif i == 5:
            variable = V_Z
            y_label = "z-position [m]"
            y_tics = np.linspace(0, 7, num=8)
        elif i == 6:
            variable = V_Z
            y_label = "z-position error[m]"
            y_tics = np.linspace(0.0, 1.6, num=5)
        elif i == 7:
            variable = V_YAW
            y_label = "yaw-rotation [deg]"
            y_tics = np.linspace(0, 160, num=5)
        elif i == 8:
            variable = V_YAW
            y_label = "yaw-rotation error [deg]"
            y_tics = np.linspace(0, 160, num=5)

        # Label x-axis
        if i==7 or i==8:
            ax.set_xlabel('Time [s]')

        # Plot the estimate values
        if i % 2 == 1:
            for j in range(4):
                if (i==7 and j==1): # Skip the ellipse yaw estimate
                   continue 

                index = index_values[j]
                color = color_values[j]
                legend_text = legend_values[j]
            
                data = stored_array[:, index:index+6][:,variable]
            
                time_stamps_local = time_stamps.copy()
                time_stamps_local[np.isnan(data)] = np.nan
        
                line, = ax.plot(time_stamps_local, data)
                line.set_color(color)

                if i == 1:
                    legend_lines.append(line)
            
        # Plot the estimate errors
        if i % 2 == 0:
            ax.axhline(y=0, color='grey', linestyle='--') # Plot the zero-line

            for j in range(3):
                if (i==8 and j==0): # Skip the ellipse yaw estimate
                    continue
                index = index_errors[j]
                color = color_errors[j]            

                data = stored_array[:, index:index+6][:,variable]
            
                time_stamps_local = time_stamps.copy()
                time_stamps_local[np.isnan(data)] = np.nan
        
                line, = ax.plot(time_stamps_local, data)
                line.set_color(color)
            
        ax.set_ylabel(y_label)
        ax.set_yticks(y_tics)


    fig.legend(legend_lines, legend_values, bbox_to_anchor=(0.3, 0.6, 0.4, 0.77), loc='center', ncol=4)

    fig.tight_layout(w_pad=1.5, rect=[0, 0, 1, 0.98])

    folder = './plots/'
    title = 'Up_down_5m'
    plt.savefig(folder+title+'.svg')

    # fig.draw
    # plt.waitforbuttonpress(0)
    # plt.close()

    # plt.show()

    
def plot_hover_compare(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m):
    file_titles = ['Hover_x', 'Hover_y', 'Hover_z', 'None', 'None', 'Hover_yaw']
    titles = ['z=0.5m','z=1m','z=2.0m','z=3m','z=5m','z=10m']
    all_hover_data = np.array([hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m])

    index_values = [1, 7, 13, 19]
    color_values = ['green', 'blue', 'red', 'orange']
    legend_values = ['ground truth', 'ellipse', 'arrow', 'corners']
    y_labels = ['x-position [m]', 'y-position [m]', 'z-position [m]', 'none', 'none', 'yaw-rotation [m]']

    x_ytics = [np.linspace(-0.10, 0.20, num=7)]*6
    y_ytics = [np.linspace(-0.15, 0.15, num=7)]*6
    z_ytics = [np.linspace(0.30, 0.70, num=5), np.linspace(0.80, 1.20, num=5), np.linspace(1.70, 2.10, num=5),
        np.linspace(2.80, 3.20, num=5), np.linspace(4.80, 5.80, num=6), np.linspace(9.80, 10.20, num=5)]
    yaw_ytics = [np.linspace(-8, 8, num=9), np.linspace(-8, 8, num=9), np.linspace(-8, 8, num=9),
        np.linspace(-8, 8, num=9), np.linspace(0, 140, num=8), np.linspace(-8, 8, num=9)]

    for variable in [V_X, V_Y, V_Z, V_YAW]:
        file_title = file_titles[variable]
        y_label = y_labels[variable]
        fig = plt.figure(figsize=(10,12))

        for i in range(6):
            ax = plt.subplot(3,2,i+1)
            if i==4 or i==5:
                ax.set_xlabel('Time [s]')

            title = titles[i]
            hover_data = all_hover_data[i]
            time_stamps = hover_data[:, 0]

            plt.title(title)
            plt.grid()
            plt.xlim(time_stamps[0], time_stamps[-1])

            for j in range(4):
                if variable==V_YAW and j==1: # Skip the ellipse yaw estimate
                    continue
                index = index_values[j]
                color = color_values[j]
                legend_text = legend_values[j]
            
                data = hover_data[:, index:index+6][:,variable]
            
                time_stamps_local = time_stamps.copy()
                time_stamps_local[np.isnan(data)] = np.nan
        
                line, = ax.plot(time_stamps_local, data)
                line.set_color(color)
                line.set_label(legend_text if i==1 else "_nolegend_")
            
            ax.set_ylabel(y_label)

            if variable == V_X:
                ax.set_yticks(x_ytics[i])
            elif variable == V_Y:
                ax.set_yticks(y_ytics[i])
            elif variable == V_Z:
                ax.set_yticks(z_ytics[i]) 
            elif variable == V_YAW:
                ax.set_yticks(yaw_ytics[i])

            # Plot legend on top
            if i==1:
                plt.legend(bbox_to_anchor=(-0.7, 1.2, 1.2, 0.0), loc='upper right', ncol=5, mode='expand')

        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')


def plot_hover_error_compare(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m):
    file_titles = ['Hover_error_x', 'Hover_error_y', 'Hover_error_z', 'None', 'None', 'Hover_error_yaw']
    titles = ['z=0.5m','z=1m','z=2.0m','z=3m','z=5m','z=10m']
    all_hover_data = np.array([hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m])
    
    index_values = [31, 37, 43]
    color_values = ['blue', 'red', 'orange']
    legend_values = ['ellipse', 'arrow', 'corners']
    y_labels = ['x-position error[m]', 'y-position error[m]', 'z-position error[m]', 'none', 'none', 'yaw-rotation error[m]']

    x_ytics = [np.linspace(-0.15, 0.15, num=7)]*6
    # x_ytics[5] = np.linspace(-0.15, 0.15, num=7)
    y_ytics = x_ytics

    z_ytics = [np.linspace(-0.20, 0.20, num=9)]*6
    z_ytics[4] = np.linspace(0, 1.0, num=6)

    yaw_ytics = [np.linspace(-6, 6, num=7)]*6
    yaw_ytics[4] = np.linspace(-140, 140, num=15)

    for variable in [V_X, V_Y, V_Z, V_YAW]:
        file_title = file_titles[variable]
        y_label = y_labels[variable]
        fig = plt.figure(figsize=(10,12))

        for i in range(6):
            ax = plt.subplot(3,2,i+1)
            ax.axhline(y=0, color='grey', linestyle='--') # Plot the zero-line
            if i==4 or i==5:
                ax.set_xlabel('Time [s]')

            title = titles[i]
            hover_data = all_hover_data[i]
            time_stamps = hover_data[:, 0]

            plt.title(title)
            plt.grid()
            plt.xlim(time_stamps[0], time_stamps[-1])

            for j in range(3):
                if variable==V_YAW and j==0: # Skip the ellipse yaw estimate
                    continue
                index = index_values[j]
                color = color_values[j]
                legend_text = legend_values[j]
            
                data = hover_data[:, index:index+6][:,variable]
            
                time_stamps_local = time_stamps.copy()
                time_stamps_local[np.isnan(data)] = np.nan
        
                line, = ax.plot(time_stamps_local, data)
                line.set_color(color)
                line.set_label(legend_text if i==1 else "_nolegend_")
            
            ax.set_ylabel(y_label)

            if variable == V_X:
                ax.set_yticks(x_ytics[i])
            elif variable == V_Y:
                ax.set_yticks(y_ytics[i])
            elif variable == V_Z:
                ax.set_yticks(z_ytics[i]) 
            elif variable == V_YAW:
                ax.set_yticks(yaw_ytics[i])

            # Plot legend on top
            if i==1:
                plt.legend(bbox_to_anchor=(-0.5, 1.2, 0.8, 0.0), loc='upper right', ncol=5, mode='expand')

        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')



def plot_step_z(data_step_z):
    file_titles = ['Step_x', 'None', 'Step_z']
    y_labels = ['x-position [m]', 'None', 'z-position [m]']

    variables = [V_X, V_Z]
    index_values = [1, 7, 13, 19, 55] #, 25]
    color_values = ['green', 'blue', 'red', 'orange', 'grey'] #, 'black']
    legend_values = ['ground truth', 'ellipse', 'arrow', 'corners', 'filtered estimate'] #, 'dead reckoning']


    time_stamps = data_step_z[:, 0]

    for variable in variables:
        file_title = file_titles[variable]
        y_label = y_labels[variable]
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot()
        plt.grid()
        plt.xlim(time_stamps[0], time_stamps[-1])

        for i in range(len(index_values)):
            if i==0:
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(y_label)

            index = index_values[i]
            color = color_values[i]
            legend_text = legend_values[i]
        
            data = data_step_z[:, index:index+6][:,variable]
        
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = ax.plot(time_stamps_local, data)
            line.set_color(color)
            line.set_label(legend_text)

        plt.legend()
        
        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')


def plot_dead_reckoning_test(data_dead_reckoning_test):
    file_title = 'Dead_reckoning_x'

    variables = [V_X]
    index_values = [1, 55, 25]
    color_values = ['green', 'grey', 'black']
    legend_values = ['ground truth', 'filtered estimate', 'dead reckoning']

    time_stamps = data_dead_reckoning_test[:, 0]

    for variable in variables:
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot()
        plt.grid()
        plt.xlim(time_stamps[0], time_stamps[-1])

        for i in range(len(index_values)):
            if i==0:
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('x-position [m]')

            index = index_values[i]
            color = color_values[i]
            legend_text = legend_values[i]
        
            data = data_dead_reckoning_test[:, index:index+6][:,variable]
        
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = ax.plot(time_stamps_local, data)
            line.set_color(color)
            line.set_label(legend_text)

        plt.legend()
        
        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')


def plot_landing(data_landing):
    """
    Generates a 3D plot of the trajectory while landing,
    and three 2D plots from each side of the 3D plot,
    and a combined plot
    """
    ###########
    # 3D plot #
    ###########
    file_title = 'Landing_3D'
    fig = plt.figure()
    ax_3D = plt.axes(projection='3d')

    indices = [1, 55, 25]
    colors = ['green', 'grey', 'black']
    legends = ['ground truth', 'filtered estimate', 'dead reckoning']

    for i in range(len(indices)):
        index = indices[i]
        color = colors[i]
        legend = legends[i]

        data_x = data_landing[:, index:index+6][:,V_X]
        data_y = data_landing[:, index:index+6][:,V_Y]
        data_z = data_landing[:, index:index+6][:,V_Z]

        line, = ax_3D.plot3D(data_x, data_y, data_z, color)
        line.set_label(legend)
        ax_3D.legend()



        ax_3D.set_xlabel('x-position [m]')
        ax_3D.set_ylabel('y-position [m]')
        ax_3D.set_zlabel('z-position [m]')

    ax_3D.view_init(20, 190) # Set view angle for the 3D plot

    fig.tight_layout()

    folder = './plots/'
    plt.savefig(folder+file_title+'.svg')

    ############
    # 2D plots #
    ############
    file_title = 'Landing_2D'
    fig = plt.figure(figsize=(7, 12))

    axes = [plt.subplot(3,1,1), plt.subplot(3,1,2), plt.subplot(3,1,3)]
    for i in range(len(indices)):
        index = indices[i]
        color = colors[i]
        legend = legends[i]

        data_x = data_landing[:, index:index+6][:,V_X]
        data_y = data_landing[:, index:index+6][:,V_Y]
        data_z = data_landing[:, index:index+6][:,V_Z]

        # Plot xy
        ax = axes[0]
        ax.grid()
        line, = ax.plot(data_x, data_y, color)
        line.set_label(legend)
        ax.legend()
        ax.set_xlabel('x-position [m]')
        ax.set_ylabel('y-position [m]')

        # Plot xz
        ax = axes[1]
        ax.grid()
        line, = ax.plot(data_x, data_z, color)
        line.set_label(legend)
        ax.legend()
        ax.set_xlabel('x-position [m]')
        ax.set_ylabel('z-position [m]')

        # Plot yz
        ax = axes[2]
        ax.grid()
        line, = ax.plot(data_y, data_z, color)
        line.set_label(legend)
        ax.legend()
        ax.set_xlabel('y-position [m]')
        ax.set_ylabel('z-position [m]')

    fig.tight_layout()

    folder = './plots/'
    plt.savefig(folder+file_title+'.svg')

    #################
    # Combine plots #
    #################
    file_title = 'Landing_combined'
    fig = plt.figure(figsize=(10,12))
    widths = [1.5, 1]
    heights = [1, 1.5, 1]
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=widths, height_ratios=heights)

    ax_3D = fig.add_subplot(gs[1, 0], projection='3d')

    indices = [1, 55, 25]
    colors = ['green', 'grey', 'black']
    legends = ['ground truth', 'filtered estimate', 'dead reckoning']
    legend_lines = []

    for i in range(len(indices)):
        index = indices[i]
        color = colors[i]
        legend = legends[i]

        data_x = data_landing[:, index:index+6][:,V_X]
        data_y = data_landing[:, index:index+6][:,V_Y]
        data_z = data_landing[:, index:index+6][:,V_Z]

        line, = ax_3D.plot3D(data_x, data_y, data_z, color)
        # line.set_label(legend)
        legend_lines.append(line)
        ax_3D.set_xlabel('x-position [m]')
        ax_3D.set_ylabel('y-position [m]')
        ax_3D.set_zlabel('z-position [m]')

    ax_3D.view_init(20, 190) # Set view angle for the 3D plot

    # 2D plots
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[2, 0])]
    for i in range(len(indices)):
        index = indices[i]
        color = colors[i]
        legend = legends[i]

        data_x = data_landing[:, index:index+6][:,V_X]
        data_y = data_landing[:, index:index+6][:,V_Y]
        data_z = data_landing[:, index:index+6][:,V_Z]

        # Plot xy
        ax = axes[0]
        ax.grid()
        ax.plot(data_y, data_x, color)

        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")

        ax.set_xlabel('y-position [m]')
        ax.set_ylabel('x-position [m]')

        xlim_left, xlim_right = ax.get_xlim()   # Invert x-axis
        ax.set_xlim(xlim_right, xlim_left)

        ax.set_aspect(1.5)

        # Plot xz
        ax = axes[1]
        ax.grid()
        ax.plot(data_x, data_z, color)
        ax.set_xlabel('x-position [m]')
        ax.set_ylabel('z-position [m]')

        ax.set_aspect(0.2)

        # Plot yz
        ax = axes[2]
        ax.grid()
        ax.plot(data_y, data_z, color)
        ax.set_xlabel('y-position [m]')
        ax.set_ylabel('z-position [m]')

        xlim_left, xlim_right = ax.get_xlim()   # Invert x-axis
        ax.set_xlim(xlim_right, xlim_left)

        ax.set_aspect(0.4)


    fig.legend(legend_lines, legends, bbox_to_anchor=(0.57, 0.47, 0.4, 0.77), loc='center')

    fig.tight_layout()

    folder = './plots/'
    plt.savefig(folder+file_title+'.svg')


def plot_yaw_test(data_yaw_test):
    file_title = 'Yaw_test'

    variables = [V_YAW]
    index_values = [1, 13, 19, 55, 25]
    color_values = ['green', 'red', 'orange', 'grey', 'black']
    legend_values = ['ground truth', 'arrow', 'corners', 'filtered estimate', 'dead reckoning']

    time_stamps = data_yaw_test[:, 0]

    for variable in variables:
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot()
        plt.grid()
        plt.xlim(time_stamps[0], time_stamps[-1])

        for i in range(len(index_values)):
            if i==0:
                ax.set_xlabel('Time [s]')
                ax.set_ylabel('yaw-rotation [deg]')

            index = index_values[i]
            color = color_values[i]
            legend_text = legend_values[i]
        
            data = data_yaw_test[:, index:index+6][:,variable]
        
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = ax.plot(time_stamps_local, data)
            line.set_color(color)
            line.set_label(legend_text)

        plt.legend()
        
        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')


def plot_hold_hover(data_hold_hover_test):
    file_titles = ['Hover_hold_x', 'Hover_hold_y', 'Hover_hold_z']
    y_labels = ['x-position [m]', 'y-position [m]', 'z-position [m]']

    variables = [V_X, V_Y, V_Z]
    index_values = [7, 13, 19]#, 55, 25]
    color_values = ['blue', 'red', 'orange']#, 'grey', 'black']
    legend_values = ['ellipse', 'arrow', 'corners']#, 'filtered estimate', 'dead reckoning']


    time_stamps = data_hold_hover_test[:, 0]

    for variable in variables:
        file_title = file_titles[variable]
        y_label = y_labels[variable]
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot()
        plt.grid()
        plt.xlim(time_stamps[0], time_stamps[-1])

        for i in range(len(index_values)):
            if i==0:
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(y_label)

            index = index_values[i]
            color = color_values[i]
            legend_text = legend_values[i]
        
            data = data_hold_hover_test[:, index:index+6][:,variable]
        
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = ax.plot(time_stamps_local, data)
            line.set_color(color)
            line.set_label(legend_text)

        plt.legend()
        
        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')


def plot_outside_flight(data_outside_test):
    file_titles = ['Outdoor_x', 'Outdoor_y', 'Outdoor_z']
    y_labels = ['x-position [m]', 'y-position [m]', 'z-position [m]']

    variables = [V_X, V_Y, V_Z]
    index_values = [7, 13, 19]#, 55, 25]
    color_values = ['blue', 'red', 'orange']#, 'grey', 'black']
    legend_values = ['ellipse', 'arrow', 'corners']#, 'filtered estimate', 'dead reckoning']


    time_stamps = data_outside_test[:, 0]

    for variable in variables:
        file_title = file_titles[variable]
        y_label = y_labels[variable]
        fig = plt.figure(figsize=(7,5))
        ax = plt.subplot()
        plt.grid()
        plt.xlim(time_stamps[0], 100)

        for i in range(len(index_values)):
            if i==0:
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(y_label)

            index = index_values[i]
            color = color_values[i]
            legend_text = legend_values[i]
        
            data = data_outside_test[:, index:index+6][:,variable]
        
            time_stamps_local = time_stamps.copy()
            time_stamps_local[np.isnan(data)] = np.nan

            line, = ax.plot(time_stamps_local, data)
            line.set_color(color)
            line.set_label(legend_text)

        plt.legend()
        
        fig.tight_layout()

        folder = './plots/'
        plt.savefig(folder+file_title+'.svg')



# List if indices
## 0    time_stamps

## 1    ground_truth

## 7    est_ellipse
## 13	est_arrow
## 19	est_corners
## 25	est_dead_reckoning

## 31	est_error_ellipse
## 37	est_error_arrow
## 43	est_error_corners
## 49	est_error_dead_reckoning

## 55   filtered_estimate
def calculate_accuracy(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m):
    variables = [V_X, V_Y, V_Z, V_YAW]
    variable_headings = ['x-position', 'y-position', 'z-position', 'yaw-rotation']
    variable_units = [' [mm]', ' [mm]', ' [mm]', ' [deg]']
    
    heights = [0.5 , 1.0, 2.0, 3.0, 5.0, 10.0]
    data_heights = [hover_0_5m , hover_1m, hover_2m, hover_3m, hover_5m, hover_10m]

    index_methods = [31, 37, 43]
    legend_methods = ['ellipse', 'arrow', 'corners']

    for index_v in range(len(variables)):
        heading = variable_headings[index_v]
        print "+-" + '-'*len(heading) + '-+'
        print "| " + heading + ' |'
        print "+-" + '-'*len(heading) + '-+' 
        variable = variables[index_v]
        variable_unit = variable_units[index_v]

        header = "| {:>6}  | {:>8} | {:>11} {:>11} {:>11} {:>11} {:>7} |".format('Height', 'Method', 'Min'+variable_unit, 'Max'+variable_unit, 'Mean'+variable_unit, 'Std'+variable_unit, 'Avbl.')
        print "+"+"-"* (len(header)-2)+"+"
        print header
        print "+"+"-"* (len(header)-2)+"+"
        for index_h in range(len(heights)):
            height = heights[index_h]
            data = data_heights[index_h]

            for index_m in range(len(index_methods)):
                index_method = index_methods[index_m]
                legend_method = legend_methods[index_m]

                data_method = data[:, index_method:index_method+6][:,variable]
                number_of_data_points = len(data_method)
                is_not_none = np.count_nonzero(~np.isnan(data_method))
                availability = is_not_none / float(number_of_data_points) * 100 # to percentage

                if variable == V_X or variable == V_Y or variable == V_Z:
                    convertion = 1000 # convert from m to mm
                else:
                    convertion = 1
                data_method = data_method[~np.isnan(data_method)]*convertion # Remove nan values and perform convertion
                
                if availability != 0 and not (variable == V_YAW and legend_method == 'ellipse'):
                    min_error = np.min(data_method)
                    max_error = np.max(data_method)

                    mean = np.mean(data_method)
                    std = np.std(data_method)
                    print "| {:>6.1f}m | {:>8} | {:>11.2f} {:>11.2f} {:>11.2f} {:>11.2f} {:>6.0f}% |".format(height, legend_method, min_error, max_error, mean, std, availability)
                else:
                    # min_error, max_error, mean, std = np.nan, np.nan, np.nan, np.nan
                    min_error, max_error, mean, std = '-', '-', '-', '-'
                    print "| {:>6.1f}m | {:>8} | {:>11} {:>11} {:>11} {:>11} {:>6.0f}% |".format(height, legend_method, min_error, max_error, mean, std, availability)

            
            print "+"+"-"* (len(header)-2)+"+"


def main():
    # Load the data
    # folder = './catkin_ws/src/uav_vision/data_storage/'
    folder = './catkin_ws/src/uav_vision/data_storage/experiment_data/'
    
    # # Up and down test
    # test_number = 1
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # up_and_down_5m = np.load(path, allow_pickle=True)

    # # Hover tests
    # test_number = 2
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_0_5m = np.load(path, allow_pickle=True)

    # test_number = 3
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_1m = np.load(path, allow_pickle=True)

    # test_number = 4
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_2m = np.load(path, allow_pickle=True)

    # test_number = 5
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_3m = np.load(path, allow_pickle=True)

    # test_number = 6
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_5m = np.load(path, allow_pickle=True)

    # test_number = 7
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # hover_10m = np.load(path, allow_pickle=True)

    # # Step test
    # test_number = 8
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # data_step_z = np.load(path, allow_pickle=True)

    # # Dead reckoning test
    # test_number = 9
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # data_dead_reckoning_test = np.load(path, allow_pickle=True)

    # # Land test
    # test_number = 10
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # data_landing = np.load(path, allow_pickle=True)

    # # Yaw test
    # test_number = 11
    # filename = 'test_'+str(test_number)+'.npy'
    # path = folder + filename
    # data_yaw_test = np.load(path, allow_pickle=True)

    # Hold hover test
    test_number = 24
    filename = 'test_'+str(test_number)+'.npy'
    path = folder + filename
    data_hold_hover_test = np.load(path, allow_pickle=True)

    # Outdoor test
    test_number = 33
    filename = 'test_'+str(test_number)+'.npy'
    path = folder + filename
    data_outside_test = np.load(path, allow_pickle=True)
    

    #################
    # Plot the data #
    #################
    # plot_data_manually(up_and_down_5m)

    # plot_hover_compare(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m)
    
    # plot_hover_error_compare(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m)

    # plot_step_z(data_step_z)

    # plot_dead_reckoning_test(data_dead_reckoning_test)

    # plot_landing(data_landing)

    # plot_yaw_test(data_yaw_test)

    plot_hold_hover(data_hold_hover_test)
    
    

    plot_outside_flight(data_outside_test)
    

    #####################
    # Calculate on data #
    #####################
    # calculate_accuracy(hover_0_5m, hover_1m, hover_2m, hover_3m, hover_5m, hover_10m)


if __name__ == '__main__':
    main()