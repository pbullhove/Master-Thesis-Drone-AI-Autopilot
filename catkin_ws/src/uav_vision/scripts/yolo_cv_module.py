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
    best = max(matches, key=lambda x: x.probability)
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

def est_rotation(H, Arrow):
    # rotation defined as positive right of line between H and arrow.
    x_h, y_h = est_center_of_bb(H)
    x_a, y_a = est_center_of_bb(Arrow)
    theta = math.atan2(y_a-y_h, x_a-x_h)
    theta -= math.pi/2
    theta *= -1
    return theta

def downscale_H_by_rotation(H, rotation):
    cx, cy = est_center_of_bb(H)
    theta = int(rad2deg(rotation))
    theta = theta % 180
    theta = abs(theta)

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


def estimate_center_rotation_and_radius(bounding_boxes):
    H_bb_radius_scale_factor = 2.60
    # rospy.loginfo(bounding_boxes)
    classes = list(item.Class for item in bounding_boxes)
    # rospy.loginfo(classes)
    center = [None, None]
    radius = None
    rotation = None
    if 'H' in classes:
        H = find_best_bb_of_class(bounding_boxes, 'H')
        if 'Arrow' in classes:
            Arrow = find_best_bb_of_class(bounding_boxes, 'Arrow')
            rotation = est_rotation(H, Arrow)
            H = downscale_H_by_rotation(H, rotation)
        center = est_center_of_bb(H)
        radius = H_bb_radius_scale_factor*est_radius_of_bb(H)

    else:
        if 'Helipad' in classes:
            Helipad = find_best_bb_of_class(bounding_boxes, 'Helipad')
            center = est_center_of_bb(Helipad)
            radius = est_radius_of_bb(Helipad)

    rospy.loginfo('\ncenter: %s \nradius %s\nrotation: %s', center, radius, rotation)
    return center, radius, rotation


def main():
    rospy.init_node('yolo_cv_module', anonymous=True)

    rospy.Subscriber('/drone_ground_truth', Twist, gt_callback)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, bb_callback)

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
            current_pose_estimate = transform_pixel_position_to_world_coordinates(center_px, radius_px)
            current_pose_estimate.angular.z = rotation if rotation is not None else current_pose_estimate.angular.z
            pub_est.publish(current_pose_estimate)

        current_error = calculate_estimation_error(current_pose_estimate, current_ground_truth)
        if current_error is not None:
            pub_error.publish(current_error)

        rate.sleep()

if __name__ == '__main__':
    main()
