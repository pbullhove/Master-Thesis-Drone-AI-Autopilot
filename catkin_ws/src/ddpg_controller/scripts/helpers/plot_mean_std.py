import matplotlib.pyplot as plt
import numpy as np
import json
plt.style.use('seaborn-whitegrid')

def extract_good_data(states, start_x, start_y, start_z):
    x = []
    y = []
    z = []

    for i in range(len(states)):
        # want to plot error? take abs and subtract goal from value: error = abs(state - goal)
        x.append(states[i][0])
        y.append(states[i][1])
        z.append(states[i][2])

    x = x[start_x:]
    y = y[start_y:]
    z = z[start_z:]
    z = [x - 2 for x in z]  # subtract 2 because setpoint is 2 but we want error

    return x, y, z

################ TEST 1
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may5_2/may5_2_plots/test1.json') as json_file:
    states = json.load(json_file)

x1, y1, z1 = extract_good_data(states, 18, 20, 20)

print("test1")
print(np.mean(x1))
print(np.std(x1))
print(np.mean(y1))
print(np.std(y1))
print(np.mean(z1))
print(np.std(z1))


################ TEST 2
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may5_2/may5_2_plots/test2.json') as json_file:
    states = json.load(json_file)

x2, y2, z2 = extract_good_data(states, 20, 20, 20)

print("test2")
print(np.mean(x2))
print(np.std(x2))
print(np.mean(y2))
print(np.std(y2))
print(np.mean(z2))
print(np.std(z2))

################ TEST 3
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may5_2/may5_2_plots/test3.json') as json_file:
    states = json.load(json_file)

x3, y3, z3 = extract_good_data(states, 28, 28, 15)

print("test3")
print(np.mean(x3))
print(np.std(x3))
print(np.mean(y3))
print(np.std(y3))
print(np.mean(z3))
print(np.std(z3))

################ TEST 4
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may5_2/may5_2_plots/test4.json') as json_file:
    states = json.load(json_file)

x4, y4, z4 = extract_good_data(states, 20, 20, 20)

print("test4")
print(np.mean(x4))
print(np.std(x4))
print(np.mean(y4))
print(np.std(y4))
print(np.mean(z4))
print(np.std(z4))


################ TEST 5
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may5_2/may5_2_plots/test5.json') as json_file:
    states = json.load(json_file)

x5, y5, z5 = extract_good_data(states, 0, 0, 12)

print("test5")
print(np.mean(x5))
print(np.std(x5))
print(np.mean(y5))
print(np.std(y5))
print(np.mean(z5))
print(np.std(z5))

##### merge

args = (x1, x2, x3, x4, x5)
x = np.concatenate(args)
args = (y1, y2, y3, y4, y5)
y = np.concatenate(args)
args = (z1, z2, z3, z4, z5)
z = np.concatenate(args)


min_len = min([len(x), len(y), len(z)])
print(min_len)

pos_error_array = []
# get total positional error. 162 was the shortest array among x y and z
for i in range(min_len):
    pos_error_array.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.errorbar(1, np.mean(x), np.std(x), fmt="o")
plt.errorbar(2, np.mean(y), np.std(y), fmt="o")
plt.errorbar(3, np.mean(z), np.std(z), fmt="o")
plt.errorbar(4, np.mean(pos_error_array), np.std(pos_error_array), fmt="o")



text_values = [r"$x-p_x$", r"$y-p_y$", r"$z-p_z$", r"$||\boldsymbol{\Tilde{x}}||$"]
x_values = np.arange(1, len(text_values) + 1, 1)
plt.xticks(x_values, text_values)
plt.ylabel("Positional error [m]")
plt.show()
