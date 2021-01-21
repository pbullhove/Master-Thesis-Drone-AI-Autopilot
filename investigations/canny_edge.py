import cv2
import numpy as np

filename = '3_green_ellipse.png'
green_ellipse = cv2.imread(filename)

edges = cv2.Canny(green_ellipse,100,200)
# edges = cv2.Canny(image=image, first_threshold=100, second_threshold=200, apertureSize = 3)
result = np.where(edges == 255)


# bgr = cv2.cvtColor(green_ellipse, cv2.COLOR_GRAY2BGR)
marked = green_ellipse.copy()

marked[result] = [0, 0, 255]

cv2.imwrite('ellipse_edges.png', marked)
