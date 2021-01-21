import numpy as np
import json
import matplotlib.pyplot as plt
import math

def Average(mylist):
    if len(mylist) == 1:
        return mylist
    return sum(mylist) / len(mylist)


with open('/home/danie/catkin_ws/src/ddpg/src/results/apr16_3/critic_loss_array.json') as json_file:
    critic_loss_array = json.load(json_file)


avg_q_value_array = []
avgnumber = 1 #number of elements to average over

# print(int(math.floor(len(clean_reward_array)/avgnumber)))

for i in range(int(round(len(critic_loss_array)/avgnumber))-1):
    if avgnumber == 1:
        current_list = [critic_loss_array[i]]
    else:
        current_list = critic_loss_array[i*avgnumber:(i+1)*avgnumber-1]
    # print(current_list)
    avg_q_value_array.append(Average(current_list))

fig = plt.figure(2)
ylabeltext = "Average reward per %i episodes" % avgnumber
# ylabeltext = "Average reward per episode"
plt.plot(avg_q_value_array)
plt.ylabel(ylabeltext)
plt.xlabel("Episodes")
plt.show()