import math

def deg2rad(deg):
    return deg*math.pi/180

def rad2deg(rad):
    return rad*180/math.pi

def est_size_of_bb_given_H(H):
    w, h = H
    H_scaling = []
    with open('H_rotation_table.txt', 'w+') as f:
        for angle in range(91):
            angle_rad = deg2rad(angle)
            cos = abs(math.cos(angle_rad))
            sin = abs(math.sin(angle_rad))
            w_bb =  cos*w + sin*h
            h_bb = sin*w + cos*h
            sw = round(w/w_bb, 3)
            sh = round(h/h_bb, 3)
            H_scaling.append((sw,sh))
            f.write(f'{sw};{sh}\n')
            #f.write(f'({w_bb/w:.3f},{h_bb/h:.3f}),')

    print(H_scaling)

est_size_of_bb_given_H((2,3))
