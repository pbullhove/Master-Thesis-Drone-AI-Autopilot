import numpy as np
import json
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

# with open('/home/danie/catkin_ws/src/ddpg/src/results/may12_2/may5_2_plots/test1.json') as json_file:
#with open('/home/danie/catkin_ws/src/ddpg/src/test8.json') as json_file:
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test8.json') as json_file:
    states = json.load(json_file)

# if results come from descend agent, include yaw 
descend_agent = True


x = []
y = []
z = []
if descend_agent:
    rotz = []

# fixed bounds of for so it is flexible to several episodes in state recorder file. only plots last episode
for i in range(len(states)-50, len(states)):
    # want to plot error? take abs and subtract goal from value: error = abs(state - goal)
    x.append(states[i][0])
    y.append(states[i][1])
    z.append(states[i][2])
    rotz.append(states[i][3] * np.pi/180)

fig = plt.figure(1)
plt.grid()
if descend_agent:
    ylabeltext = "Pose" 
else:
    ylabeltext = "Position" 



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# ylabeltext = "Average reward per episode"
plt.plot(x, 'r', label=r'$x$')
plt.plot(y, 'b', label=r'$y$')
plt.plot(z, 'g', label=r'$z$')
plt.hlines(0, 0, 49, 'r', '-')
plt.hlines(0, 0, 49, 'b', '--')
if descend_agent:
    plt.hlines(0.3, 0, 49, 'g', 'dashed') # for z target
    plt.plot(rotz, 'm', label=r"$\psi$") # NOTE: change to latex format?
    plt.hlines(0, 0, 49, 'm', '--')
else:
    plt.hlines(2, 0, 49, 'g', 'dashed')

plt.ylabel(ylabeltext)
plt.xlabel("Step")
plt.legend(loc="best")
plt.show()

if not descend_agent:
    fig = plt.figure(2)
    plt.grid(linestyle=':')
    plt.ylim(-2.5, 2.5)
    plt.xlim(-2.5, 2.5)
    ylabeltext = "y"
    # ylabeltext = "Average reward per episode"
    plt.plot(x, y, color="k")
    plt.plot(0, 0, marker="+", color="red", markersize=20, mew=2)

    # for 0,0,3, i make a red-green cross
    #plt.plot(0, 0, marker=0, color="red",   markersize=10, mew=2)
    #plt.plot(0, 0, marker=2, color="red",   markersize=10, mew=2)
    #plt.plot(0, 0, marker=1, color="green", markersize=10, mew=2)
    #plt.plot(0, 0, marker=3, color="green", markersize=10, mew=2)


    plt.plot(x[0], y[0], marker="+", color="green", markersize=20, mew=2)
    plt.ylabel(ylabeltext)
    plt.xlabel("x")
    plt.show()



    fig = plt.figure(3)
    ax = fig.gca(projection='3d')
    ax.set_xlim3d(-2.0, 2.0)
    ax.set_ylim3d(-2.0, 2.0)
    ax.set_zlim3d(1, 3.0)
    ax.plot([x[0]], [y[0]], [z[0]], marker="+", color="green", markersize=20, mew=2)
    ax.plot([0], [0], [2], marker="+", color="red", markersize=20, mew=2)

    # Data for a three-dimensional line
    ax.plot(x, y, z)
    plt.show()