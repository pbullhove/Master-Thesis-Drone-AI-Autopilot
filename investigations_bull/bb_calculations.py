import math
import numpy

REAL_HELIPAD_RADIUS = 0.80

# Dimentions from investigations/design_helipad.py
D_H_LONG = 9.0
D_H_SHORT = 3.0
D_RADIUS = 32.0
D_ARROW = 23.0

factor = REAL_HELIPAD_RADIUS / D_RADIUS
D_RADIUS = REAL_HELIPAD_RADIUS
D_H_LONG /= factor
D_H_SHORT /= factor
D_ARROW /= factor


IMG_SIZE = (416,416)

def rad_to_deg(rad):
    return rad*180/math.pi

def deg_to_rad(deg):
    return deg*math.pi/180

def est_yaw_angle_rad(helipad_center, arrow_location):
    # yaw defined as positive right of line between H and arrow.
    x1,y1 = helipad_center
    x2,y2 = arrow_location
    angle = math.atan2(y2-y1, x2-x1)
    angle -= math.pi/2
    return angle

def convert_YOLO_format_to_pixels(bb, img_size=IMG_SIZE):
    x1, y1, x2, y2 = bb
    assert x1 <= x2 and y1 <= y2
    assert (0 <= x1 <= 1) and (0 <= x2 <= 1)
    assert (0 <= y1 <= 1) and (0 <= y2 <= 1)
    x1 *=img_size[0]
    x2 *=img_size[0]
    y1 *= img_size[1]
    y2 *= img_size[1]
    return (x1, y1, x2, y2)

def est_center_of_bb(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center = [numpy.average((x1,x2)), numpy.average((y1,y2))]

    width = x2 - x1
    height = y2 - y1
    try:
        width_height_relationship = width/height
    except ZeroDivisionError:
        width_height_relationship = 1
    whr_lower_bound = 0.8
    whr_upper_bound = 1.2
    edge_touch_rad = 0.01

    if width_height_relationship > whr_upper_bound:
        # Part of image is out of bounds in y-direction.
        if y1 < edge_touch_rad and not y2 > 1 - edge_touch_rad:
            # BB is touching bottom side and not top side
            center[1] -= (width - height)/2
            center[1] = max(0,center[1])

        elif y2 > 1 - edge_touch_rad and not y1 < edge_touch_rad:
            # BB is touching top side and not bottom side.
            center[1] += (width - height)/2
            center[1] = min(1, center[1])

    if width_height_relationship < whr_lower_bound:
        # part of image is out of bound in x-direction:
        if x1 < edge_touch_rad and not x2 > 1 - edge_touch_rad:
            # BB is touching left side of image and not right side.
            center[0] -= (height - width)/2
            center[0] = max(0,center[0])

        elif x2 > 1 - edge_touch_rad and not x1 < edge_touch_rad:
            # BB is touching right side of image and not left side.
            center[0] += (width - height)/2
            center[0] = min(1, center[0])

    return tuple(center)

def est_size_of_H(H_bounding_box, angle):
    # https://stackoverflow.com/questions/9971230/calculate-rotated-rectangle-size-from-known-bounding-box-coordinates
    x1, y1, x2, y2 = H_bounding_box
    bx = x2-x1
    by = y2-y1
    cos = math.cos(angle)
    sin = math.sin(angle)

    H_est_width =  bx*cos - by*sin
    H_est_height = -bx*sin + by*cos


    return H_est_width, H_est_height



def test_est_center_of_bb():
    print('Init Test:     test_est_center_of_bb')
    bb = (0,0,1,1)
    center = est_center_of_bb(bb)
    expected_center = (0.5, 0.5)
    assert center == expected_center, f"Expected center at {expected_center}, got center at {center}"

    bb = (0,0,0.5,0.5)
    center = est_center_of_bb(bb)
    expected_center = (0.25, 0.25)
    assert center == expected_center, f"Expected center at {expected_center}, got center at {center}"

    bb = (0.5,0.5,0.5,0.5)
    center = est_center_of_bb(bb)
    expected_center = (0.5, 0.5)
    assert center == expected_center, f"Expected center at {expected_center}, got center at {center}"

    bb = (0.20,0.20,0.6,0.8)
    center = est_center_of_bb(bb)
    expected_center = (0.4, 0.5)
    assert center == expected_center, f"Expected center at {expected_center}, got center at {center}"

    bb = (0,0.4,0.8,1)
    center = est_center_of_bb(bb)
    expected_center = (0.4, 0.8)
    assert center == expected_center, f"Expected center at {expected_center}, got center at {center}"
    print('TEST SUCCESS:  test_est_center_of_bb()')

def test_est_size_of_H():
    bb = (0,0,0.1,0.2)
    size_bb = (0.1,0.2)
    angle = deg_to_rad(45)
    expected_size = (size_bb[0]*math.cos(angle) + size_bb[1]*math.sin(angle),
                    size_bb[0]*math.sin(angle) + size_bb[1]*math.cos(angle))
    size = est_size_of_H(bb, angle)
    assert size == expected_size, f"Expected H size {expected_size}, got size {size}"


def test_convert_YOLO_format_to_pixels():
    print('Init Test:     test_est_center_of_bb')
    yolo = (0,0,1,1)
    pixels = convert_YOLO_format_to_pixels(yolo)
    assert pixels == (0,0,IMG_SIZE[0], IMG_SIZE[1])

    yolo = (0,0,0.5,0.5)
    pixels = convert_YOLO_format_to_pixels(yolo)
    assert pixels == tuple(map(int,(0,0,IMG_SIZE[0]/2,IMG_SIZE[1]/2)))

    yolo = (0.5,0.5,0.5,0.5)
    pixels = convert_YOLO_format_to_pixels(yolo)
    assert pixels == tuple(map(int,(IMG_SIZE[0]/2, IMG_SIZE[1]/2, IMG_SIZE[0]/2, IMG_SIZE[1]/2)))

    yolo = (0.5,0.5,0.75,1)
    pixels = convert_YOLO_format_to_pixels(yolo)
    assert pixels == tuple(map(int,(IMG_SIZE[0]/2, IMG_SIZE[1]/2, IMG_SIZE[0]*3/4, IMG_SIZE[1])))

    print('TEST SUCCESS:  test_convert_YOLO_format_to_pixels()')

def test_est_yaw_angle_rad():
    pass



def main():
    # test_convert_YOLO_format_to_pixels()
    # test_est_center_of_bb()
    # test_est_size_of_H()
    # test_est_yaw_angle_rad()
    est_size_of_bb_given_H((1,2))


if __name__ == "__main__":
    main()
