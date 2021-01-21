

min_x = 10000
max_x = -10000
min_y = 10000
max_y = -10000
min_z = 10000
max_z = -10000
min_vx = 10000
max_vx = -10000
min_vy = 10000
max_vy = -10000
min_vz = 10000
max_vz = -10000
min_ax = 10000
max_ax = -10000
min_ay = 10000
max_ay = -10000
min_az = 10000
max_az = -10000
min_rotx = 10000
max_rotx = -10000
min_roty = 10000
max_roty = -10000
min_rotz = 10000
max_rotz = -10000
min_wx = 10000
max_wx = -10000
min_wy = 10000
max_wy = -10000
min_wz = 10000
max_wz = -10000

def update_max_and_min(state):
    global min_x, max_x, min_y, max_y, min_z, max_z, min_vx, max_vx, min_vy, max_vy, min_vz, max_vz, min_ax, \
        max_ax, min_ay, max_ay, min_az, max_az, min_rotx, max_rotx, min_roty, max_roty, min_rotz, max_rotz, min_wx, max_wx, min_wy, max_wy, min_wz, max_wz
    # xyz
    if state[0] < min_x:
        min_x = state[0]
    if state[0] > max_x:
        max_x = state[0]
    if state[1] < min_y:
        min_y = state[1]
    if state[1] > max_y:
        max_y = state[1]
    if state[2] < min_z:
        min_z = state[2]
    if state[2] > max_z:
        max_z = state[2]

    # vx vy vz
    if state[3] < min_vx:
        min_vx = state[3]
    if state[3] > max_vx:
        max_vx = state[3]
    if state[4] < min_vy:
        min_vy = state[4]
    if state[4] > max_vy:
        max_vy = state[4]
    if state[5] < min_vz:
        min_vz = state[5]
    if state[5] > max_vz:
        max_vz = state[5]

    # ax ay az
    if state[6] < min_ax:
        min_ax = state[6]
    if state[6] > max_ax:
        max_ax = state[6]
    if state[7] < min_ay:
        min_ay = state[7]
    if state[7] > max_ay:
        max_ay = state[7]
    if state[8] < min_az:
        min_az = state[8]
    if state[8] > max_az:
        max_az = state[8]

    # rotx roty rotz
    if state[9] < min_rotx:
        min_rotx = state[9]
    if state[9] > max_rotx:
        max_rotx = state[9]
    if state[10] < min_roty:
        min_roty = state[10]
    if state[10] > max_roty:
        max_roty = state[10]
    if state[11] < min_rotz:
        min_rotz = state[11]
    if state[11] > max_rotz:
        max_rotz = state[11]

    # wz wy wz
    if state[12] < min_wx:
        min_wx = state[12]
    if state[12] > max_wx:
        max_wx = state[12]
    if state[13] < min_wy:
        min_wy = state[13]
    if state[13] > max_wy:
        max_wy = state[13]
    if state[14] < min_wz:
        min_wz = state[14]
    if state[14] > max_wz:
        max_wz = state[14]

    maxminlist = [min_x, max_x, min_y, max_y, min_z, max_z, min_vx, max_vx, min_vy, max_vy, min_vz, max_vz, min_ax, \
        max_ax, min_ay, max_ay, min_az, max_az, min_rotx, max_rotx, min_roty, max_roty, min_rotz, max_rotz, min_wx, max_wx, min_wy, max_wy, min_wz, max_wz]

    return maxminlist
