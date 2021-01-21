import numpy as np
import cv2
import math



# Dimentions

# D_H_SHORT = 3.0
# D_H_LONG = 9.0
# D_ARROW = 30.0
# D_RADIUS = 40.0

# SIZE_ARROW = 4.0
# SIZE_ORANGE_CIRCLE = 2.0

D_H_SHORT = 3.0
D_H_LONG = 9.0
D_ARROW = 25.0
D_RADIUS = 32.0

SIZE_ARROW = 4.0
SIZE_ORANGE_CIRCLE = 1.5


RADIUS_PX = 1200.0

IMG_SIZE = np.int0(RADIUS_PX*2+1)

IMG_CENTER = (np.int0(RADIUS_PX), np.int0(RADIUS_PX))


def hsv_to_opencv_hsv(hue, saturation, value):
    """ 
    Function that takes in hue, saturation and value in the ranges
        hue: [0, 360] degrees,  saturation: [0, 100] %,     value: [0, 100] %
    and converts it to OpenCV hsv which operates with the ranges
        hue: [0, 180],          saturation: [0, 255],       value: [0, 255]
    """
    converting_constant = np.array([0.5, 2.55, 2.55]) 
    return np.array([ hue, saturation, value])*converting_constant


HSV_GREEN_COLOR = hsv_to_opencv_hsv(120, 100, 40)
HSV_LIGHT_ORANGE_COLOR = hsv_to_opencv_hsv(35, 100, 100)
HSV_WHITE_COLOR = hsv_to_opencv_hsv(0, 0, 100)
HSV_BLUE_COLOR = hsv_to_opencv_hsv(240, 60, 70)

HSV_GRAY_COLOR = hsv_to_opencv_hsv(0, 0, 50)


HSV_BACKGROUND = HSV_WHITE_COLOR



def hsv_save_image(image, label='helipad', is_gray=False):
    # folder = 'image_processing/detect_h/'
    folder = 'image_processing/helipad_design/'
    if is_gray:
        cv2.imwrite(folder+label+".png", image)
    else:
        cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))
    return image

def to_px_value(relative_value):
    return np.int0(relative_value * RADIUS_PX / D_RADIUS)


def draw_background(img):
    img[:] = HSV_BACKGROUND


def create_mask_circle():
    mask_circle = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)    

    radius = np.int0(RADIUS_PX)

    center = IMG_CENTER
    radius = radius
    color = 255
    thickness = -1 # Fill

    return cv2.circle(mask_circle, center, radius, color, thickness)


def draw_green_circle(img):
    radius = np.int0(RADIUS_PX)

    center = IMG_CENTER
    radius = radius
    color = HSV_GREEN_COLOR
    thickness = -1 # Fill

    cv2.circle(img, center, radius, color, thickness)


def draw_orange_circle(img):
    radius = to_px_value(D_ARROW-SIZE_ARROW)

    center = IMG_CENTER
    radius = radius
    color = HSV_LIGHT_ORANGE_COLOR
    thickness = to_px_value(SIZE_ORANGE_CIRCLE)

    cv2.circle(img, center, radius, color, thickness)


def draw_arrowhead(img):
    px_arrow = to_px_value(D_ARROW)

    px_arrow_size = to_px_value(SIZE_ARROW)

    pt_1 = np.array([IMG_CENTER[0] , IMG_CENTER[0] - px_arrow])
    pt_2 = pt_1 + np.array([px_arrow_size, px_arrow_size])
    pt_3 = pt_1 + np.array([-px_arrow_size, px_arrow_size])

    pts_array = np.array([pt_1, pt_2, pt_3]).reshape((-1, 1, 2))

    cv2.fillPoly(img, [pts_array], HSV_LIGHT_ORANGE_COLOR, 8)


def draw_white_rectangle(img, center_x, center_y, dist_x, dist_y, color):
    """ Input in pixels """
    pt1 = (center_x - dist_x, center_y - dist_y)
    pt2 = (center_x + dist_x, center_y + dist_y)

    cv2.rectangle(img, pt1, pt2, color, -1)


def draw_h(img):

    # Draw cross-bar
    center_x, center_y = IMG_CENTER
    dist_x = to_px_value(D_H_LONG / 2.0)
    dist_y = to_px_value(D_H_SHORT / 2.0)
    draw_white_rectangle(img, center_x, center_y, dist_x, dist_y, HSV_WHITE_COLOR)

    # Measure for the legs
    leg_dist_x = to_px_value(D_H_SHORT / 2.0)
    leg_dist_y = to_px_value((D_H_LONG*2 + D_H_SHORT) / 2.0)
    leg_center_y = center_y
    
    # Draw left leg
    left_leg_center_x = center_x - to_px_value((D_H_LONG + D_H_SHORT) / 2.0)
    draw_white_rectangle(img, left_leg_center_x, leg_center_y, leg_dist_x, leg_dist_y, HSV_WHITE_COLOR)
    
    # Draw right leg
    right_leg_center_x = center_x + to_px_value((D_H_LONG + D_H_SHORT) / 2.0)
    draw_white_rectangle(img, right_leg_center_x, leg_center_y, leg_dist_x, leg_dist_y, HSV_WHITE_COLOR)


def rotate_image(img):
    center = (IMG_CENTER[0], IMG_CENTER[1])
    angle = 45
    scale = 1

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST)

    mask_circle = create_mask_circle()
    hsv_save_image(mask_circle, "mask_circle", is_gray=True)

    _ , mask_binary = cv2.threshold(mask_circle, 128, 255, cv2.THRESH_BINARY)
    
    rotated[mask_binary != 255] = HSV_BACKGROUND
    
    return rotated


def make_helipad():
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)    
    large_canvas = np.zeros((4969, 4969, 3), dtype=np.uint8)   

    draw_background(canvas)
    draw_green_circle(canvas)
    draw_orange_circle(canvas)
    draw_arrowhead(canvas)
    draw_h(canvas)

    large_canvas[:,:,:] = HSV_GRAY_COLOR

    large_canvas[2529:4930,42:2443,] = canvas

    hsv_save_image(canvas)
    hsv_save_image(large_canvas, "texture_helipad")
    
    # Rotate the image
    rotated = rotate_image(canvas)

    hsv_save_image(rotated, "helipad_rotated")


make_helipad()