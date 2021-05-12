#!/usr/bin/env python

"""



"""


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


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)



def est_rotation(center, Arrow):
    """
    Estimates the quadcopter yaw rotation given the estimated center of the Helipad
    as well as the estimated center of the arrow. Quadcopter rotation is defined
    with respect to world frame coordinate axis.
    yaw=0 is when arrow is pointing at three-o-clock, and positively increasing when
        arrow is rotating the same direction as clock arm movement.
    y is defined 0 in top of the image, and increases positively downwards.

    input:
        center: np.array[2] - [x,y] pixel coordinates
        Arrow: np.array[2] - [x,y] pixel coordinates

    output:
        degs: float - estimated yaw angle of the quadcopter
    """
    dy = center[1] - Arrow[1]
    dx = Arrow[0] - center[0]
    rads = math.atan2(dy,dx)
    degs = rads*180 / math.pi
    degs *= -1
    return degs


def is_good_bb(bb):
    """
    Returns true for bounding boxes that are within the desired proportions,
    them being relatively square.
    Bbs that have one side being 5 times or longer than the other are discarded.

    input:
        bb: yolo bounding box

    output.
        discard or not: bool
    """
    bb_w = bb.xmax - bb.xmin
    bb_h = bb.ymax - bb.ymin
    if 0.2 > bb_w/bb_h > 5:
        return False
    else:
        return True


def estimate_center_rotation_and_radius(bounding_boxes):
    """
    Function which estimates the center and radius of the helipad in the camera
    frame, using YOLO BoundingBox of predicted locations of Helipad and Arrow from the camera frame.
    Estimates radius as 0.97*sidelength of Helipad BoundingBox.
    Estimates center as center of Helipad BoundingBox
    Estimates yaw using est_rotation() with center and center of Arrow BoundingBox.

    input:
        bounding_boxes: [n]bouding_box - list of yolo bouding_boxes
    output:
        center: [int, int] - estimated pixel coordinates of center of landing platform.
        radius: int - estimated pixel length of radius of landing platform
        rotation: float degrees - estimated yaw of quadcopter
    """
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
    return center, radius, rotation


def main():
    rospy.init_node('dnn_CV_module', anonymous=True)

    rospy.Subscriber('/drone_ground_truth', Twist, gt_callback)
    rospy.Subscriber('/darknet_ros/bounding_boxes', BoundingBoxes, bb_callback)
    rospy.Subscriber('/filtered_estimate', Twist, filtered_estimate_callback)
    pub_est = rospy.Publisher("/estimate/dnnCV", Twist, queue_size=10)
    pub_ground_truth = rospy.Publisher('/drone_ground_truth', Twist, queue_size=10)
    pub_error = rospy.Publisher("/estimate_error/dnn_error", Twist, queue_size=10)
    pub_center_radius = rospy.Publisher("/results/dnn_error", Twist, queue_size=10)

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
