#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from darknet_ros_msgs.msg import BoundingBox
from darknet_ros_msgs.msg import BoundingBoxes

import numpy as np
import math
import config as cfg

global_is_simulator = cfg.is_simulator

if global_is_simulator:
    camera_offset_x = 150 # mm
    camera_offset_z = -45 # mm
else:
    camera_offset_x = -60 # mm
    camera_offset_z = -45 # mm

IMG_WIDTH = 640
IMG_HEIGHT = 360

global_ground_truth = None
def gt_callback(data):
    global global_ground_truth
    global_ground_truth = data
    # global_ground_truth = np.array([data.linear.x, data.linear.y, data.linear.z, 0, 0, data.angular.z])


global_bounding_boxes = None
def bb_callback(data):
    global global_bounding_boxes
    global_bounding_boxes = data

filtered_estimate = None
def filtered_estimate_callback(data):
    global filtered_estimate
    filtered_estimate = data

def rad2deg(rad):
    return rad*180/math.pi

def deg2rad(deg):
    return deg*math.pi/180

def get_test_bounding_boxes():

    Helipad = BoundingBox()
    Helipad.probability = 0.5
    Helipad.xmin = 312
    Helipad.ymin = 120
    Helipad.xmax = 337
    Helipad.ymax = 148
    Helipad.id = 2
    Helipad.Class = "Helipad"

    H = BoundingBox()
    H.probability = 0.5
    H.xmin = 320
    H.ymin = 128
    H.xmax = 330
    H.ymax = 138
    H.id = 0
    H.Class = "H"

    Arrow = BoundingBox()
    Arrow.probability = 0.5
    Arrow.xmin = 333
    Arrow.ymin = 140
    Arrow.xmax = 335
    Arrow.ymax = 143
    Arrow.id = 1
    Arrow.Class = "Arrow"

    bbs = BoundingBoxes()
    bbs.bounding_boxes = [Helipad, H, Arrow]
    return bbs


def transform_pixel_position_to_world_coordinates(center_px, radius_px):
    center_px = (center_px[1], center_px[0]) # such that x = height, y = width for this
    focal_length = 374.67
    real_radius = 390 # mm (780mm in diameter / 2)

    # Center of image
    x_0 = IMG_HEIGHT/2.0
    y_0 = IMG_WIDTH/2.0

    # Find distances from center of image to center of LP
    d_x = x_0 - center_px[0]
    d_y = y_0 - center_px[1]

    est_z = real_radius*focal_length / radius_px

    # Camera is placed 150 mm along x-axis of the drone
    # Since the camera is pointing down, the x and y axis of the drone
    # is the inverse of the x and y axis of the camera
    est_x = -((est_z * d_x / focal_length) + camera_offset_x) # mm adjustment for translated camera frame in x direction
    est_y = -(est_z * d_y / focal_length)
    est_z += camera_offset_z # mm adjustment for translated camera frame in z direction

    est = Twist()
    est.linear.x = est_x / 1000.0
    est.linear.y = est_y / 1000.0
    est.linear.z = est_z / 1000.0

    return est


def calculate_estimation_error(est, gt):
    if gt == None:
        return None
    est = Twist() if est is None else est
    error = Twist()
    error.linear.x = est.linear.x - gt.linear.x
    error.linear.y = est.linear.y - gt.linear.y
    error.linear.z = est.linear.z - gt.linear.z
    error.angular.x = est.angular.x - gt.angular.x
    error.angular.y = est.angular.y - gt.angular.y
    error.angular.z = est.angular.z - gt.angular.z
    return error


def find_best_bb_of_class(bounding_boxes, classname):
    matches =  list(item for item in bounding_boxes if item.Class == classname)
    try:
        best = max(matches, key=lambda x: x.probability)
    except ValueError as e:
        best = None
    return best

def est_center_of_bb(bb):
    width = bb.xmax - bb.xmin
    height = bb.ymax - bb.ymin
    center = [bb.xmin + width/2.0 ,bb.ymin + height/2.0]
    map(int, center)
    return center

def est_radius_of_bb(bb):
    width = bb.xmax - bb.xmin
    height = bb.ymax - bb.ymin
    radius = (width + height)/4
    return radius



def downscale_H_by_rotation(H):
    global filtered_estimate
    theta = abs(filtered_estimate.angular.z)
    rotation = deg2rad(theta)
    cx, cy = est_center_of_bb(H)

    cos = abs(math.cos(rotation))
    sin = abs(math.sin(rotation))
    scaling_width = (cos*2 + sin*3)/2
    scaling_height = (sin*2 + cos*3)/3

    width = H.xmax - H.xmin
    height = H.ymax - H.ymin
    new_width = width * scaling_width
    new_height = height * scaling_height

    H.xmin = cx - int(new_width/2)
    H.ymin = cy - int(new_height/2)
    H.xmax = cx + int(new_width/2)
    H.ymax = cy + int(new_height/2)

    return H

def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def calc_angle_between_vectors(vector_1, vector_2):
    v1_x = vector_1[0]
    v1_y = vector_1[1]

    v2_x = vector_2[0]
    v2_y = vector_2[1]

    return np.arctan2( v1_x*v2_y - v1_y*v2_x, v1_x*v2_x + v1_y*v2_y)




def est_rotation(center, Arrow):
    # if center == [None, None]:
    #     return None
    # arrow_vector = np.array(np.array(est_center_of_bb(Arrow)) - np.array(center))
    # arrow_unit_vector = normalize_vector(arrow_vector)
    # arrow_unit_vector_yx = np.array([arrow_unit_vector[1], arrow_unit_vector[0]])
    # rad = calc_angle_between_vectors(arrow_unit_vector_yx, np.array([0,1]))
    # deg = rad2deg(rad)
    print("center: ", center)
    print("arrow: ", Arrow)

    dy = center[1] - Arrow[1]
    dx = Arrow[0] - center[0]
    rads = math.atan2(dy,dx)
    degs = rads*180 / math.pi
    degs *= -1
    return degs


def is_good_bb(bb):
    bb_w = bb.xmax - bb.xmin
    bb_h = bb.ymax - bb.ymin
    if 0.2 > bb_w/bb_h > 5:
        return False
    else:
        return True


def estimate_center_rotation_and_radius(bounding_boxes):
    H_bb_radius_scale_factor = 2.60
    # rospy.loginfo(bounding_boxes)
    bounding_boxes = [bb for bb in bounding_boxes if is_good_bb(bb)]
    classes = list(item.Class for item in bounding_boxes)
    # rospy.loginfo(classes)
    center = [None, None]
    radius = None
    rotation = None

    Helipad = find_best_bb_of_class(bounding_boxes, 'Helipad')
    H = find_best_bb_of_class(bounding_boxes, 'H')
    Arrow = find_best_bb_of_class(bounding_boxes, 'Arrow')


    if Helipad != None:
        center = est_center_of_bb(Helipad)
        radius = 0.97*est_radius_of_bb(Helipad)
        if Arrow != None:
            rotation = est_rotation(center, est_center_of_bb(Arrow))

    rospy.loginfo('\ncenter: %s \nradius %s\nrotation: %s', center, radius, rotation)
    return center, radius, rotation


def main():
    rospy.init_node('yolo_cv_module', anonymous=True)

    rospy.Subscriber('/drone_ground_truth', Twist, gt_callback)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, bb_callback)
    rospy.Subscriber('/filtered_estimate', Twist, filtered_estimate_callback)
    pub_est = rospy.Publisher("/estimate/yolo_estimate", Twist, queue_size=10)
    pub_ground_truth = rospy.Publisher('/drone_ground_truth', Twist, queue_size=10)
    pub_error = rospy.Publisher("/estimate_error/yolo_error", Twist, queue_size=10)
    pub_center_radius = rospy.Publisher("/results/yolo_error", Twist, queue_size=10)

    est_pose_msg = Twist()

    rospy.loginfo("Starting yolo_CV module")
    # if not global_is_simulator:
    #     global_ground_truth = np.zeros(6)

    use_test_bbs = 0
    previous_bounding_boxes = None
    current_pose_estimate = None
    count = 0
    rate = rospy.Rate(10) # Hz
    while not rospy.is_shutdown():
        current_ground_truth = global_ground_truth # Fetch the latest ground truth pose available
        if use_test_bbs:
            current_bounding_boxes = get_test_bounding_boxes()
        else:
            current_bounding_boxes = global_bounding_boxes
        if (current_bounding_boxes is not None) and (current_bounding_boxes != previous_bounding_boxes):
            previous_bounding_boxes = current_bounding_boxes
            center_px, radius_px, rotation = estimate_center_rotation_and_radius(current_bounding_boxes.bounding_boxes)
            # rospy.loginfo('center_px: %s,  radius_px: %s,  rotation: %s', center_px, radius_px, rotation)
            if all(center_px) and radius_px:
                current_pose_estimate = transform_pixel_position_to_world_coordinates(center_px, radius_px)
                current_pose_estimate.angular.z = rotation if rotation is not None else 0.0
                pub_est.publish(current_pose_estimate)

        current_error = calculate_estimation_error(current_pose_estimate, current_ground_truth)
        if current_error is not None:
            pub_error.publish(current_error)

        rate.sleep()

if __name__ == '__main__':
    main()
