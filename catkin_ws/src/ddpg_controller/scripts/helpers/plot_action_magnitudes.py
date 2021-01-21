import numpy as np
import json
import matplotlib.pyplot as plt
import math


with open('/home/danie/catkin_ws/src/ddpg/src/results/apr25/replay_buffer.json') as json_file:
#with open('/home/danie/catkin_ws/src/ddpg/src/replay_buffer.json') as json_file:
    replay_buffer = json.load(json_file)

print("len", len(replay_buffer))
# print(replay_buffer[0])
# print(replay_buffer[0][1])

actions_x = []
actions_y = []
actions_z = []

for i in range(len(replay_buffer)):
    actions_x.append(replay_buffer[i][1][0])
    actions_y.append(replay_buffer[i][1][1])
    actions_z.append(replay_buffer[i][1][2])

print("actions_x[-15:]", actions_x[-15:])

fig = plt.figure(2)
ylabeltext = "Action magnitude"
plt.plot(actions_y[-15:])
plt.ylabel(ylabeltext)
plt.xlabel("Step")
plt.show()