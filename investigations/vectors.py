import numpy as np

a = np.array([1060, 3751])
b = np.array([1059, 3672])
d = np.array([1401, 3751])
arrow = np.array([1231, 2849])
center = np.array([1231, 3712])
border = np.array([1231, 2602])

l_1 = np.linalg.norm(b-a)
l_2 = np.linalg.norm(a-d)
l_3 = np.linalg.norm(arrow-center)
l_4 = np.linalg.norm(border-center)

print "l_1:", l_1
print "l_2:", l_2
print "l_3:", l_3
print "l_4:", l_4