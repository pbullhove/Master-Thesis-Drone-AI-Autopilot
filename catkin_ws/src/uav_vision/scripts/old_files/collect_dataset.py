#!/usr/bin/env python
import rospy
import numpy as np
import json
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R

from std_msgs.msg import Empty
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

import config as cfg


bridge = CvBridge()

global_gt_pose = None
global_image = None
save_images = False
global_signal = False

IMG_WIDTH = 640
IMG_HEIGHT = 360


def signal_callback(data):
    global global_signal
    global_signal = True


def gt_callback(data):
    global global_gt_pose
    global_gt_pose = data.pose.pose


def image_callback(data):
    global global_image

    try:
        global_image = bridge.imgmsg_to_cv2(data, 'bgr8') # {'bgr8' or 'rgb8}
    except CvBridgeError as e:
        rospy.loginfo(e)


def get_relative_position(gt_pose):
    # Transform ground truth in body frame wrt. world frame to body frame wrt. landing platform

    ##########
    # 0 -> 2 #
    ##########

    # Position
    p_x = gt_pose.position.x
    p_y = gt_pose.position.y
    p_z = gt_pose.position.z

    # Translation of the world frame to body frame wrt. the world frame
    d_0_2 = np.array([p_x, p_y, p_z])

    # Orientation
    q_x = gt_pose.orientation.x
    q_y = gt_pose.orientation.y
    q_z = gt_pose.orientation.z
    q_w = gt_pose.orientation.w

    # Rotation of the body frame wrt. the world frame
    r_0_2 = R.from_quat([q_x, q_y, q_z, q_w])
    r_2_0 = r_0_2.inv()
    

    ##########
    # 0 -> 1 #
    ##########
    
    # Translation of the world frame to landing frame wrt. the world frame
    offset_x = 1.0
    offset_y = 0.0
    offset_z = 0.54
    d_0_1 = np.array([offset_x, offset_y, offset_z])

    # Rotation of the world frame to landing frame wrt. the world frame
    # r_0_1 = np.identity(3) # No rotation, only translation
    r_0_1 = np.identity(3) # np.linalg.inv(r_0_1)


    ##########
    # 2 -> 1 #
    ##########
    # Transformation of the body frame to landing frame wrt. the body frame
    
    # Translation of the landing frame to bdy frame wrt. the landing frame
    d_1_2 = d_0_2 - d_0_1

    # Rotation of the body frame to landing frame wrt. the body frame
    r_2_1 = r_2_0

    yaw = r_2_1.as_euler('xyz')[2]
    r_2_1_yaw = R.from_euler('z', yaw)

    # Translation of the body frame to landing frame wrt. the body frame
    # Only yaw rotation is considered
    d_2_1 = -r_2_1_yaw.apply(d_1_2)

    # Translation of the landing frame to body frame wrt. the body frame
    # This is more intuitive for the controller
    d_2_1_inv = -d_2_1

    return d_2_1_inv


# Computer Vision
def make_blurry(image, blurr):
    return cv2.medianBlur(image, blurr)


def hsv_make_orange_to_green(hsv):
    bgr_green = np.uint8([[[30,90,30]]])
    hsv_green = cv2.cvtColor(bgr_green,cv2.COLOR_BGR2HSV)

    lower_orange = np.array([8,128,64])
    upper_orange = np.array([29,255,255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # change the orange to green
    imask = mask>0
    orange_to_green = hsv.copy()
    orange_to_green[imask] = hsv_green

    return orange_to_green


def hsv_find_green_mask(hsv):
    bgr_green = np.uint8([[[30,90,30]]])
    hsv_green = cv2.cvtColor(bgr_green,cv2.COLOR_BGR2HSV)

    lower_green_h = 45
    lower_green_s = 50 * 0.01*255
    lower_green_v = 25 * 0.01*255

    upper_green_h = 75
    upper_green_s = 100 * 0.01*255
    upper_green_v = 70 * 0.01*255

    lower_green = np.array([lower_green_h,lower_green_s,lower_green_v])
    upper_green = np.array([upper_green_h,upper_green_s,upper_green_v])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
  
    # keep only the green
    imask = green_mask>0
    green = np.zeros_like(hsv, np.uint8)
    green[imask] = hsv_green
  
    return green


def save_image(image, label='image', is_gray=False):
    if save_images:
        folder = '/home/thomas/Desktop/image_processing/'
        if is_gray:
            cv2.imwrite(folder+label+".png", image)
        else:
            cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

    return image


def flood_fill(img):
    h,w,chn = img.shape
    seed = (w/2,h/2)
    seed = (0,0)

    mask = np.zeros((h+2,w+2),np.uint8) # Adding a padding of 1

    floodflags = 8
    # floodflags |= cv2.FLOODFILL_FIXED_RANGE
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    num,img,mask,rect = cv2.floodFill(img, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
    mask = mask[1:h+1,1:w+1] # Removing the padding

    return mask


def preprocessing(raw_img):
    hsv = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)
    hsv = save_image(hsv, '1_hsv')
    
    hsv = save_image(hsv_make_orange_to_green(hsv), '2_orange_to_green')
    hsv = save_image(make_blurry(hsv, 3), '3_make_blurry')
    hsv = save_image(hsv_find_green_mask(hsv), '4_green_mask')
    hsv = save_image(make_blurry(hsv, 3), '5_make_blurry')
    gray = save_image(flood_fill(hsv), '6_flood_fill', is_gray=True)

    return gray


def fit_ellipse(points):
    x = points[1]
    y = IMG_HEIGHT-points[0]

    D11 = np.square(x)
    D12 = x*y
    D13 = np.square(y)
    D1 = np.array([D11, D12, D13]).T
    D2 = np.array([x, y, np.ones(x.shape[0])]).T

    S1 = np.dot(D1.T,D1)
    S2 = np.dot(D1.T,D2)
    S3 = np.dot(D2.T,D2)
    inv_S3 = np.linalg.inv(S3)

    T = - np.dot(inv_S3, S2.T) # for getting a2 from a1

    M = S1 + np.dot(S2, T)

    C1 = np.array([
        [0, 0, 0.5],
        [0, -1, 0],
        [0.5, 0, 0]
    ])

    M = np.dot(C1, M) # This premultiplication can possibly be made more efficient
    
    eigenvalues, eigenvectors = np.linalg.eig(M)
    cond = 4*eigenvectors[0]*eigenvectors[2] - np.square(eigenvectors[0])
    a1 = eigenvectors[:,cond > 0]
    
    # Choose the first if there are two eigenvectors with cond > 0
    # NB! I am not sure if this is always correct
    if a1.shape[1] > 1:
        a1 = np.array([a1[:,0]]).T

    a = np.concatenate((a1, np.dot(T, a1)))[:,0] # Choose the inner column with [:,0]

    if np.any(np.iscomplex(a)):
        print("Found complex number")
        return None
    else:
        return a


def get_ellipse_parameters(raw_img):
    gray = preprocessing(raw_img)

    edges = cv2.Canny(gray,100,200)
    result = np.where(edges == 255)

    ellipse = fit_ellipse(result)

    return ellipse


def main():
    rospy.init_node('collect_dataset', anonymous=True)

    rospy.Subscriber('/ground_truth/state', Odometry, gt_callback)
    rospy.Subscriber('/ardrone/bottom/image_raw', Image, image_callback)
    rospy.Subscriber('/ardrone/takeoff', Empty, signal_callback)

    pub_heartbeat = rospy.Publisher("/heartbeat", Empty, queue_size=10)

    heartbeat_msg = Empty()

    rospy.loginfo("Starting collecting dataset")

    filename="image_above.jpg"

    dataset = {}
    filename_dataset = 'full_dataset_train_low_flight.json'

    count = 0
    NUM_DATAPOINTS = 1400

    rate = rospy.Rate(20) # Hz
    while not rospy.is_shutdown():

        if (global_gt_pose is not None) and (global_image is not None):
            pub_heartbeat.publish(heartbeat_msg)
            
            relative_position = get_relative_position(global_gt_pose)    
            # global_image = cv2.imread(filename)
            ellipse_parameters = get_ellipse_parameters(global_image)
        
            if ellipse_parameters is not None:
                data_instance = {
                    'ellipse': ellipse_parameters.tolist(),
                    'ground_truth': relative_position.tolist()
                }
                dataset[count] = data_instance

                if global_signal:
                    with open(filename_dataset, 'w') as fp:
                        json.dump(dataset, fp)
                    rospy.loginfo("Count: " + str(count))
                    rospy.loginfo("Saved as json")
                    break
                else:
                    rospy.loginfo("Count: " + str(count))
                count += 1
        else:
            rospy.loginfo("Waiting for signals")

        rate.sleep()
    
    
if __name__ == '__main__':
    main()