import matplotlib.pyplot as plt
import numpy as np
import json
plt.style.use('seaborn-whitegrid')

def extract_good_data(states, start_x, start_y, start_z, start_rotz):
    x = []
    y = []
    z = []
    rotz = []

    for i in range(len(states)-50,len(states)):
        # want to plot error? take abs and subtract goal from value: error = abs(state - goal)
        x.append(states[i][0])
        y.append(states[i][1])
        z.append(states[i][2])
        rotz.append(states[i][3])

    x = x[start_x:]
    y = y[start_y:]
    z = z[start_z:]
    z = [x - 0.3 for x in z]  # subtract 2 because setpoint is 2 but we want error
    rotz = rotz[start_rotz:]

    return x, y, z, rotz

################ TEST 1
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test1.json') as json_file:
    states = json.load(json_file)

x1, y1, z1, rotz1 = extract_good_data(states, 5, 5, 40, 5)

print("test1")
print(np.mean(x1))
print(np.std(x1))
print(np.mean(y1))
print(np.std(y1))
print(np.mean(z1))
print(np.std(z1))
print(np.mean(rotz1))
print(np.std(rotz1))


################ TEST 2
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test2.json') as json_file:
    states = json.load(json_file)

x2, y2, z2, rotz2 = extract_good_data(states, 7, 7, 38, 10)

print("test2")
print(np.mean(x2))
print(np.std(x2))
print(np.mean(y2))
print(np.std(y2))
print(np.mean(z2))
print(np.std(z2))
print(np.mean(rotz2))
print(np.std(rotz2))


################ TEST 3
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test3.json') as json_file:
    states = json.load(json_file)

x3, y3, z3, rotz3 = extract_good_data(states, 10, 10, 40, 10)

print("test3")
print(np.mean(x3))
print(np.std(x3))
print(np.mean(y3))
print(np.std(y3))
print(np.mean(z3))
print(np.std(z3))
print(np.mean(rotz3))
print(np.std(rotz3))

################ TEST 4
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test4.json') as json_file:
    states = json.load(json_file)

x4, y4, z4, rotz4 = extract_good_data(states, 10, 10, 35, 10)

print("test4")
print(np.mean(x4))
print(np.std(x4))
print(np.mean(y4))
print(np.std(y4))
print(np.mean(z4))
print(np.std(z4))
print(np.mean(rotz4))
print(np.std(rotz4))

################ TEST 5
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test5.json') as json_file:
    states = json.load(json_file)

x5, y5, z5, rotz5 = extract_good_data(states, 10, 10, 35, 10)

print("test5")
print(np.mean(x5))
print(np.std(x5))
print(np.mean(y5))
print(np.std(y5))
print(np.mean(z5))
print(np.std(z5))
print(np.mean(rotz5))
print(np.std(rotz5))

################ TEST 6
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test5.json') as json_file:
    states = json.load(json_file)

x6, y6, z6, rotz6 = extract_good_data(states, 10, 10, 40, 10)

print("test6")
print(np.mean(x6))
print(np.std(x6))
print(np.mean(y6))
print(np.std(y6))
print(np.mean(z6))
print(np.std(z6))
print(np.mean(rotz6))
print(np.std(rotz6))

################ TEST 7
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test5.json') as json_file:
    states = json.load(json_file)

x7, y7, z7, rotz7 = extract_good_data(states, 10, 10, 35, 10)

print("test7")
print(np.mean(x7))
print(np.std(x7))
print(np.mean(y7))
print(np.std(y7))
print(np.mean(z7))
print(np.std(z7))
print(np.mean(rotz7))
print(np.std(rotz7))

################ TEST 8
with open('C:/Users/danie/Desktop/thesis/ddpg/src/results/may12_2/may12_2_plots/test5.json') as json_file:
    states = json.load(json_file)

x8, y8, z8, rotz8 = extract_good_data(states, 10, 10, 40, 10)

print("test8")
print(np.mean(x8))
print(np.std(x8))
print(np.mean(y8))
print(np.std(y8))
print(np.mean(z8))
print(np.std(z8))
print(np.mean(rotz8))
print(np.std(rotz8))


##### merge

args = (x1, x2, x3, x4, x5, x6, x7, x8)
x = np.concatenate(args)
args = (y1, y2, y3, y4, y5, y6, y7, y8)
y = np.concatenate(args)
args = (z1, z2, z3, z4, z5, z6, z7, z8)
z = np.concatenate(args)
args = (rotz1, rotz2, rotz3, rotz4, rotz5, rotz6, rotz7, rotz8)
rotz = np.concatenate(args)
rotz = rotz * np.pi/180

min_len = min([len(x), len(y), len(z), len(rotz)])
print(min_len)

pos_error_array_aug = []
pos_error_array = []
# get total positional error. 162 was the shortest array among x y and z

'''
def sum_of_squares(arr):
    a = 0
    for i in range(len(arr)):
        a += arr[i]**2
    return a
sum_x = sum_of_squares(x)
sum_y = sum_of_squares(y)
sum_z = sum_of_squares(z)
sum_rotz = sum_of_squares(rotz)
'''

for i in range(min_len):
    pos_error_array.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
    pos_error_array_aug.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2 + rotz[i]**2))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.errorbar(1, np.mean(x), np.std(x), fmt="o")
plt.errorbar(2, np.mean(y), np.std(y), fmt="o")
plt.errorbar(3, np.mean(z), np.std(z), fmt="o")
plt.errorbar(4, np.mean(rotz), np.std(rotz), fmt="o")
plt.errorbar(5, np.mean(pos_error_array), np.std(pos_error_array), fmt="o")
plt.errorbar(6, np.mean(pos_error_array_aug), np.std(pos_error_array_aug), fmt="o")

plt.hlines(0, 0.5, 6.5, 'k', '--')


text_values = [r"$x-p_x$", r"$y-p_y$", r"$z-p_z$", r"$\psi-p_{\psi}$", r"$||\boldsymbol{\Tilde{x}}||$", r"$||\boldsymbol{\Tilde{x}}_a||$"]
x_values = np.arange(1, len(text_values) + 1, 1)
plt.xticks(x_values, text_values)
plt.ylabel("Mean and standard deviation")
plt.show()
