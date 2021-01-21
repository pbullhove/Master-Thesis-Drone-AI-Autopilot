import numpy as np
import json
import matplotlib.pyplot as plt
import math

def Average(mylist):
    if len(mylist) == 1:
        return mylist
    return sum(mylist) / len(mylist)


#with open('/home/danie/catkin_ws/src/ddpg/src/results/apr19_2/q_value_array.json') as json_file:
with open('/home/danie/catkin_ws/src/ddpg/src/q_value_array.json') as json_file:
    q_values = json.load(json_file)


avg_q_value_array = []
avgnumber = 1 #number of elements to average over

# print(int(math.floor(len(clean_reward_array)/avgnumber)))

for i in range(int(round(len(q_values)/avgnumber))-1):
    if avgnumber == 1:
        current_list = [q_values[i]]
    else:
        current_list = q_values[i*avgnumber:(i+1)*avgnumber-1]
    # print(current_list)
    avg_q_value_array.append(Average(current_list))

fig = plt.figure(2)
ylabeltext = "Average reward per %i episodes" % avgnumber
# ylabeltext = "Average reward per episode"
plt.plot(avg_q_value_array)
plt.ylabel(ylabeltext)
plt.xlabel("Episodes")
plt.show()