import numpy as np
import json
import matplotlib.pyplot as plt
import math

def Average(mylist):
    if len(mylist) == 1:
        return mylist
    return sum(mylist) / len(mylist)


with open('/home/danie/catkin_ws/src/ddpg/src/results/may12_2/reward_array.json') as json_file:
# with open('/home/danie/catkin_ws/src/ddpg/src/reward_array.json') as json_file:
    reward_array = json.load(json_file)

# remove all elements with exactly 0.0 in reward list
#reward_array = [i for i in reward_array if i != 0.0]

print("len", len(reward_array))

avg_reward_array = []
avgnumber = 1 #number of elements to average over

# print(int(math.floor(len(clean_reward_array)/avgnumber)))

for i in range(int(round(len(reward_array)/avgnumber))-1):
    if avgnumber == 1:
        current_list = [reward_array[i]]
    else:
        current_list = reward_array[i*avgnumber:(i+1)*avgnumber-1]
    # print(current_list)
    avg_reward_array.append(Average(current_list))

fig = plt.figure(2)
ylabeltext = "Average reward per %i episodes" % avgnumber
# ylabeltext = "Average reward per episode"
plt.plot(avg_reward_array)
plt.ylabel(ylabeltext)
plt.xlabel("Episodes")
plt.show()