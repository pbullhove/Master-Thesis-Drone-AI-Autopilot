import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import math
from scipy import ndimage

import pandas as pd


IMG_WIDTH = 640
IMG_HEIGHT = 360

def load_ellipse_from_file():
    filepath = 'ellipse.jpg'

    img = cv2.imread(filepath) # import as BGR
    winname = "image"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 40,500)  # Move it to (40,30)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print(type(img))

    ellipse = []

    for row in range(255):
        for col in range(255):
            if not np.array_equal(img[row][col], [255, 255, 255]):
                # print(img[row][col])
                ellipse.append([row, col])

    np_ellipse = np.array(ellipse)

    len_ellipse = np_ellipse.shape[0]
    print(len_ellipse)


    # id_sample = np.random.choice(len_ellipse, 10, replace=False)

    # Choose the same points every time for easier comparison
    id_sample = range(0,len_ellipse, 20)
    print(id_sample)


    np_ellipse_sample = np_ellipse[id_sample]

    print(np_ellipse_sample)

    plt.figure(figsize=(8,8))
    plt.plot(np_ellipse_sample[:,1], 255-np_ellipse_sample[:,0], 'go')
    plt.xlim((0,255))
    plt.ylim((0,255))
    plt.show()

scattered_ellipse = np.array(
    [   [ 66, 96],
        [ 70, 69],
        [ 74, 133],
        [ 82, 43],
        [ 92, 152],
        [109, 157],
        [130, 37],
        [142, 133],
        [148, 118],
        [151, 105]
    ]
)

scattered_ellipse2 = np.array(
    [   [ 50, 50],
        [ 30, 70],
        [ 70, 70],
        [150, 150],
        [130, 37],
        [142, 133]
    ]
)

# plt.figure(figsize=(8,8))
# plt.plot(scattered_ellipse[:,1], 255-scattered_ellipse[:,0], 'go')
# plt.xlim((0,255))
# plt.ylim((0,255))
# plt.show()




def fit_ellipse_from_square_img(points):
    x = points[:,1]
    y = 255-points[:,0]

    print(x.shape)

    print('x:', x)
    print('y:', y)

    # plt.figure(figsize=(8,8))
    # plt.plot(x, y, 'go')
    # plt.xlim((0,255))
    # plt.ylim((0,255))
    # plt.show()

    D11 = np.square(x)
    D12 = x*y
    D13 = np.square(y)
    D1 = np.array([D11, D12, D13]).T
    print("D1:")
    print(D1)
    print

    D2 = np.array([x, y, np.ones(x.shape[0])]).T
    print("D2:")
    print(D2)
    print

    S1 = np.dot(D1.T,D1)
    print("S1:")
    print(S1)
    print

    S2 = np.dot(D1.T,D2)
    print("S2:")
    print(S2)
    print

    S3 = np.dot(D2.T,D2)
    print("S3:")
    print(S3)
    print

    inv_S3 = np.linalg.inv(S3)
    print("inv_S3:")
    print(inv_S3)
    print

    T = - np.dot(inv_S3, S2.T) # for getting a2 from a1
    print("T:")
    print(T)
    print

    M = S1 + np.dot(S2, T)

    C1 = np.array([
        [0, 0, 0.5],
        [0, -1, 0],
        [0.5, 0, 0]
    ])

    M = np.dot(C1, M) # This premultiplication can possibly be made more efficient
    print("M:")
    print(M)
    print

    eigenvalues, eigenvectors = np.linalg.eig(M)
    print("eigenvalues:")
    print(eigenvalues)
    print
    print("eigenvectors:")
    print(eigenvectors)
    print

    cond = 4*eigenvectors[0]*eigenvectors[2] - np.square(eigenvectors[0])
    print("cond:")
    print(cond)
    print

    a1 = eigenvectors[:,cond > 0]
    print("a1:")
    print(a1)
    print

    a = np.concatenate((a1, np.dot(T, a1)))
    print("a:")
    print(a)
    print


    delta = 0.025
    xrange = np.arange(0, 255, delta)
    yrange = np.arange(0, 255, delta)
    X, Y = np.meshgrid(xrange,yrange)

    # F is one side of the equation, G is the other
    F = a[0]*X*X + a[1]*X*Y + a[2]*Y*Y + a[3]*X + a[4]*Y + a[5]
    G = 0


    plt.figure(figsize=(8,8))
    plt.plot(x, y, 'go')
    plt.xlim((0,255))
    plt.ylim((0,255))
    plt.contour(X, Y, (F - G), [0])
    plt.show()


filename = "image_above.jpg"

def fit_ellipse(points, filename="image_above.jpg"):
    x = points[1]
    y = IMG_HEIGHT-points[0]

    print(x.shape)

    print('x:', x)
    print('y:', y)

    # plt.figure(figsize=(8,8))
    # plt.plot(x, y, 'go')
    # plt.xlim((0,255))
    # plt.ylim((0,255))
    # plt.show()

    D11 = np.square(x)
    D12 = x*y
    D13 = np.square(y)
    D1 = np.array([D11, D12, D13]).T
    print("D1:")
    print(D1)
    print

    D2 = np.array([x, y, np.ones(x.shape[0])]).T
    print("D2:")
    print(D2)
    print

    S1 = np.dot(D1.T,D1)
    print("S1:")
    print(S1)
    print

    S2 = np.dot(D1.T,D2)
    print("S2:")
    print(S2)
    print

    S3 = np.dot(D2.T,D2)
    print("S3:")
    print(S3)
    print

    inv_S3 = np.linalg.inv(S3)
    print("inv_S3:")
    print(inv_S3)
    print

    T = - np.dot(inv_S3, S2.T) # for getting a2 from a1
    print("T:")
    print(T)
    print

    M = S1 + np.dot(S2, T)

    C1 = np.array([
        [0, 0, 0.5],
        [0, -1, 0],
        [0.5, 0, 0]
    ])

    M = np.dot(C1, M) # This premultiplication can possibly be made more efficient
    print("M:")
    print(M)
    print

    eigenvalues, eigenvectors = np.linalg.eig(M)
    print("eigenvalues:")
    print(eigenvalues)
    print
    print("eigenvectors:")
    print(eigenvectors)
    print

    cond = 4*eigenvectors[0]*eigenvectors[2] - np.square(eigenvectors[0])
    print("cond:")
    print(cond)
    print

    a1 = eigenvectors[:,cond > 0]

    print("a1:")
    print(a1)
    print
    print(a1.shape)
    
    # Choose the first if there are two eigenvectors with cond > 0
    # NB! I am not sure if this is always correct
    if a1.shape[1] > 1:
        a1 = np.array([a1[:,0]]).T
    print(a1.shape)
    print("a1:")
    print(a1)
    print

    a = np.concatenate((a1, np.dot(T, a1)))
    print("a:")
    print(a)
    print

    # # Drawing the ellipse
    # delta = 0.025
    # xrange = np.arange(0, IMG_WIDTH, delta)
    # yrange = np.arange(0, IMG_HEIGHT, delta)
    # X, Y = np.meshgrid(xrange,yrange)

    # # F_ellipse is one side of the equation, G_ellipse is the other
    # F_ellipse = a[0]*X*X + a[1]*X*Y + a[2]*Y*Y + a[3]*X + a[4]*Y + a[5]
    # G_ellipse = 0

    # mean_x = np.average(x)
    # mean_y = np.average(y)


    

    A = a[0][0]
    B = a[1][0]
    C = a[2][0]
    D = a[3][0]
    E = a[4][0]
    F = a[5][0]

    print('\n')
    print('a:', A)
    print('b:', B)
    print('c:', C)
    print('d:', D)
    print('e:', E)
    print('f:', F)
    print('\n')



    x_0 = (2*C*D - B*E) / (B*B - 4*A*C) 
    y_0 = (2*A*E - B*D) / (B*B - 4*A*C) 


    print("x_0 and y_0:")
    print(x_0)
    print(y_0)
    print('\n')


    inner_square = math.sqrt( (A-C)**2 + B**2)

    outside = 1 / (B**2 - 4*A*C)

    a = outside * math.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( (A+C) + inner_square))

    b = outside * math.sqrt(2*(A*E**2 + C*D**2 - B*D*E + (B**2 - 4*A*C)*F) * ( (A+C) - inner_square))

    print("a and b:")
    print(a)
    print(b)


    if B != 0:
        theta = math.atan( (1/B)*(C - A - math.sqrt( (A-C)**2 + B**2)))
    else:
        if A < C:
            theta = 0
        else:
            theta = math.pi/2
    
    print("theta:")
    print(theta)

    a_x = a*math.cos(theta)
    a_y = a*math.sin(theta)


    b_x = -b*math.sin(theta)
    b_y = b*math.cos(theta)

    # M = np.array([[a[0][0], a[1][0]/2.0],
    #     [a[1][0]/2.0, a[2][0]]])
    # # print(M)
    # eigenvalues, eigenvectors = np.linalg.eig(M)
    # lambda_1 = eigenvalues[0]
    # lambda_2 = eigenvalues[1]
    # # print(eigenvalues)
    # S = np.linalg.det(M)
    # # print(S)

    # a = -S/(lambda_1*lambda_1*lambda_2)
    # b = -S/(lambda_1*lambda_2*lambda_2)

    # print("a and b:")
    # print(a)
    # print(b)
    

    # plt.figure(figsize=(8.89,5))
    background = plt.imread(filename)
    background = np.flipud(background)
    # plt.plot(mean_x, mean_y, 'go')
    plt.imshow(background)
    plt.plot(x_0, y_0, 'bo')

    plt.plot(x_0+a_x, y_0+a_y, 'ro')
    plt.plot(x_0-a_x, y_0-a_y, 'ro')

    plt.plot(x_0+b_x, y_0+b_y, 'yo')
    plt.plot(x_0-b_x, y_0-b_y, 'yo')

    # plt.plot(x_0-j, y_0+b, 'yo')
    # plt.plot(x_0+j, y_0-b, 'yo')


    plt.xlim((0,IMG_WIDTH))
    plt.ylim((0,IMG_HEIGHT))
    # plt.contour(X, Y, (F_ellipse - G_ellipse), [0])
    plt.show()



    # u=1.     #x-position of the center
    # v=0.5    #y-position of the center
    # a=2.     #radius on the x-axis
    # b=1.5    #radius on the y-axis

    # t = np.linspace(0, 2*math.pi, 100)
    # plt.figure(figsize=(8,8))
    # plt.plot( u+a*np.cos(t) , v+b*np.sin(t) )
    # plt.xlim((0,255))
    # plt.ylim((0,255))
    # plt.grid(color='lightgray',linestyle='--')
    # plt.show()




# fit_ellipse(scattered_ellipse)
# a:
# [[-3.66860513e-01]
#  [-3.88682954e-03]
#  [-9.30267841e-01]
#  [ 6.50013809e+01]
#  [ 2.03228556e+02]
#  [-1.21953571e+04]]

def make_blurry(image, blurr):
    return cv2.medianBlur(image, blurr)


def hsv_make_orange_to_green(hsv):
    bgr_green = np.uint8([[[30,90,30]]])
    hsv_green = cv2.cvtColor(bgr_green,cv2.COLOR_BGR2HSV)

    lower_orange = np.array([8,128,64])
    upper_orange = np.array([28,255,255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # change the orange to green
    imask = mask>0
    orange_to_green = hsv.copy()
    orange_to_green[imask] = hsv_green

    return orange_to_green


def hsv_keep_orange_only(hsv):
    bgr_white = np.uint8([[[255,255,255]]])
    hsv_white = cv2.cvtColor(bgr_white,cv2.COLOR_BGR2HSV)

    bgr_black = np.uint8([[[0,0,0]]])
    hsv_black = cv2.cvtColor(bgr_black,cv2.COLOR_BGR2HSV)

    lower_orange = np.array([8,128,64])
    upper_orange = np.array([28,255,255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # change the orange to green
    imask = mask>0
    b_imask = mask<=0
    orange_to_green = hsv.copy()
    orange_to_green[imask] = hsv_white
    orange_to_green[b_imask] = hsv_black

    return orange_to_green


def hsv_find_green_mask(hsv):
    bgr_green = np.uint8([[[30,90,30]]])
    hsv_green = cv2.cvtColor(bgr_green,cv2.COLOR_BGR2HSV)

    lower_green_h = 50
    lower_green_s = 50 * 0.01*255
    lower_green_v = 25 * 0.01*255

    upper_green_h = 70
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


def hsv_make_grayscale(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray


def hsv_save_image(image, label='image', is_gray=False):
    folder = 'image_processing/'
    if is_gray:
        cv2.imwrite(folder+label+".png", image)
    else:
        cv2.imwrite(folder+label+".png", cv2.cvtColor(image, cv2.COLOR_HSV2BGR))

    return image


def load_image(filename = 'image_above.jpg'):
    img = cv2.imread(filename) # import as BGR

    # winname = "image"
    # cv2.namedWindow(winname)        # Create a named window
    # cv2.moveWindow(winname, 40,500)  # Move it to (40,30)
    # cv2.imshow(winname, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv_save_image(hsv, '1_hsv')
    # ### Preprocessing ###
    hsv = hsv_save_image(hsv_make_orange_to_green(hsv), '2_orange_to_green')
    hsv = make_blurry(hsv, 9)
    hsv = hsv_save_image(make_blurry(hsv, 3), '3_make_blurry')
    hsv = hsv_save_image(hsv_find_green_mask(hsv), '4_green_mask')
    # hsv = make_blurry(hsv, 9)
    # hsv = hsv_save_image(make_blurry(hsv, 9), '5_make_blurry_2')
    # gray = hsv_save_image(hsv_make_grayscale(hsv), '6_make_grayscale', is_gray=True)
    gray = hsv
    # gray = hsv_save_image(make_blurry(gray, 31), "6_make_blurry_3")
    # gray = hsv_save_image(make_blurry(gray, 21), '7_make_blurry_3', is_gray=True)

    return gray


def load_image_find_orange():
    filepath = 'image_above.jpg'

    img = cv2.imread(filepath) # import as BGR

    # winname = "image"
    # cv2.namedWindow(winname)        # Create a named window
    # cv2.moveWindow(winname, 40,500)  # Move it to (40,30)
    # cv2.imshow(winname, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv_save_image(hsv, '1_hsv')
    # ### Preprocessing ###
    hsv = hsv_save_image(hsv_keep_orange_only(hsv), '2_orange_to_white')
    # hsv = make_blurry(hsv, 9)
    # hsv = hsv_save_image(make_blurry(hsv, 3), '3_make_blurry')
    # hsv = hsv_save_image(hsv_find_green_mask(hsv), '4_green_mask')
    # hsv = make_blurry(hsv, 9)
    # hsv = hsv_save_image(make_blurry(hsv, 9), '5_make_blurry_2')
    gray = hsv_save_image(hsv_make_grayscale(hsv), '6_make_grayscale', is_gray=True)
    # gray = make_blurry(gray, 31)
    # gray = hsv_save_image(make_blurry(gray, 21), '7_make_blurry_3', is_gray=True)

    return gray


def direction_from_point():
    # filepath = 'image_above.jpg'
    # img = cv2.imread(filepath) # import as BGR

    point_x = 128
    point_y = 247

    background = plt.imread("image_above.jpg")
    background = np.flipud(background)
    plt.plot(point_x, point_y, 'go')
    plt.imshow(background)
    plt.xlim((0,IMG_WIDTH))
    plt.ylim((0,IMG_HEIGHT))
    plt.show()



    K = np.array([[374.6706070969281, 0.0, 320.5],
                  [0.0, 374.6706070969281, 180.5],
                  [0.0, 0.0, 1.0]
                ])

    print(K)

    # K: [374.6706070969281, 0.0, 320.5, 0.0, 374.6706070969281, 180.5, 0.0, 0.0, 1.0]


def find_contours(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create hull array for convex hull points
    hull = []
    
    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    
    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 0, 0) # blue - color for convex hull
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)

    gray = hsv_save_image(drawing, '0_hull', is_gray=True)


def ellipse_detection():
    img = load_image()

    find_contours(img)
    
    return

    edges = cv2.Canny(img,100,200)
    print(type(edges))


    unique, counts = np.unique(edges, return_counts=True)
    print(dict(zip(unique, counts)))

    result = np.where(edges == 255)
    print()
    print("Result:")
    print(result)

    fit_ellipse(result)

    x = result[1]
    y = IMG_HEIGHT - result[0]

    # print(x)

    background = plt.imread("image_above.jpg")
    background = np.flipud(background)
    plt.figure(figsize=(8.89,5))
    plt.plot(x, y, 'ro')
    plt.imshow(background)
    plt.xlim((0,IMG_WIDTH))
    plt.ylim((0,IMG_HEIGHT))
    plt.show()


    # winname = 'edges'
    # cv2.namedWindow(winname)        # Create a named window
    # cv2.moveWindow(winname, 40,500)  # Move it to (40,30)
    # cv2.imshow(winname, edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ellipse_detection()


def flood_fill(filename = "image_above.jpg"):
    im = load_image(filename)
    h,w,chn = im.shape
    seed = (w/2,h/2)
    seed = (0,0)
    # print(h, ", ", w)

    mask = np.zeros((h+2,w+2),np.uint8) # Adding a padding of 1

    floodflags = 8
    # floodflags |= cv2.FLOODFILL_FIXED_RANGE
    floodflags |= cv2.FLOODFILL_MASK_ONLY
    floodflags |= (255 << 8)

    num,im,mask,rect = cv2.floodFill(im, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)
    mask = mask[1:h+1,1:w+1] # Removing the padding


    hsv_save_image(mask, "flood_fill", is_gray=True)


    print(mask)

    # plt.imshow(im)
    # plt.show()

    # plt.imshow(mask)
    # plt.show()

    return mask



def make_dataset():
    ellipses = np.array([
        [0.1, 0.2, 0.1, 0.5, 0.1, 0.9],
        [0.2, 0.4, -0.1, 0.4, 0.5, 0.2],
        [0.5, 0.3, 0.5, -0.5, 0.3, 0.8]
    ])

    gt = np.array([
        [-1.2, -0.9, -2.1],
        [-1.0, -0.5, -2.0],
        [-0.8, -0.7, -1.9]
    ])

    dataset = np.concatenate((ellipses, gt), axis=1)
    print(dataset)

    np.savetxt("foo.csv", dataset, delimiter=",")

    # dataset = pd.DataFrame({'ellipses': ellipses, 'gt': gt})
    # print(dataset)

    # df = pd.read_csv('~/master_project/investigations/dataset.csv')



make_dataset()

# mask = flood_fill(filename)


# edges = cv2.Canny(mask,100,200)
# # print(type(edges))


# unique, counts = np.unique(edges, return_counts=True)
# print(dict(zip(unique, counts)))

# result = np.where(edges == 255)
# # print()
# # print("Result:")
# # print(result)


# fit_ellipse(result, filename)

# x = result[1]
# y = IMG_HEIGHT - result[0]

# print(x)

# background = plt.imread(filename)
# background = np.flipud(background)
# plt.figure(figsize=(8.89,5))
# plt.plot(x, y, 'ro')
# plt.imshow(background)
# plt.xlim((0,IMG_WIDTH))
# plt.ylim((0,IMG_HEIGHT))
# plt.show()