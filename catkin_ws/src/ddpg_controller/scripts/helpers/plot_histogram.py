import numpy as np
import json
import matplotlib.pyplot as plt

#with open('/home/danie/catkin_ws/src/ddpg/src/results/apr20_3/number_of_steps_array.json') as json_file:
with open('/home/danie/catkin_ws/src/ddpg/src/number_of_steps_array.json') as json_file:
    steps_array = json.load(json_file)

max_steps_found = max(steps_array)
min_steps_found = min(steps_array)

print("min", min_steps_found, "max", max_steps_found)

_ = plt.hist(steps_array, range=(min_steps_found,max_steps_found), bins=max_steps_found-min_steps_found)

plt.show()

fig = plt.figure(2)
ylabeltext = "Steps per episode"
# ylabeltext = "Average reward per episode"
plt.plot(steps_array)
plt.ylabel(ylabeltext)
plt.xlabel("Episode")
plt.show()