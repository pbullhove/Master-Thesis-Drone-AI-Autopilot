import cv2
import numpy as np

filename = '1_green_mask_sim.png'
# filename = '1_white_mask.png'
img = cv2.imread(filename)

blur_size = 9
blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_blurred = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)

block_size = 7      # It is the size of neighbourhood considered for corner detection
aperture_param = 9  # Aperture parameter of Sobel derivative used.
k_param = 0.04    # Harris detector free parameter in the equation. range: [0.04, 0.06]

block_size_original = 2
aperture_param_original = 3
k_param_original = 0.04

# find Harris corners original parameters
gray = np.float32(gray)
dst_original = cv2.cornerHarris(gray, block_size_original, aperture_param_original, k_param_original)
dst_original = cv2.dilate(dst_original,None)
ret, dst_original = cv2.threshold(dst_original,0.01*dst_original.max(),255,0)
dst_original = np.uint8(dst_original)

# find Harris corners new parameters
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, block_size, aperture_param, k_param)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find Harris corners new parameters with blur
# Blurring beforehand contributes to removing false corners. 
# It has a larger effect with images from the real quadcopter, 
# where there will be more noice due to not ideal conditions like in the smulator
gray_blurred = np.float32(gray_blurred)
dst_blur = cv2.cornerHarris(gray_blurred, block_size, aperture_param, k_param)
dst_blur = cv2.dilate(dst_blur,None)
ret, dst_blur = cv2.threshold(dst_blur,0.01*dst_blur.max(),255,0)
dst_blur = np.uint8(dst_blur)

# # find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_blur)

# define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

#####
# 1 #
#####
# Only Harris, original parameters
overlay_01 = img.copy() 
overlay_01[dst_original==255] = [0,0,255]
cv2.imwrite('cv01_original_harris.png', overlay_01)

#####
# 2 #
#####
# Only Harris, new parameters
overlay_02 = img.copy()
overlay_02[dst==255] = [0,0,255]
cv2.imwrite('cv02_new_param_harris.png', overlay_02)

#####
# 3 #
#####
# Only Harris, new parameters and blur
overlay_03 = blur.copy()
overlay_03[dst_blur==255] = [0,0,255]
cv2.imwrite('cv03_new_param_blur_harris_white.png', overlay_03)

#####
# 4 #
#####
# Now draw them
overlay_04 = blur.copy()
res = np.hstack((centroids,corners))
res = np.int0(res)
overlay_04[res[:,1],res[:,0]]=  [0,0,255]
overlay_04[res[:,3],res[:,2]] = [0,255,0]
cv2.imwrite('cv04_new_param_blur_sub_pixel_harris02.png', overlay_04)
