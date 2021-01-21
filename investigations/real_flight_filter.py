
import cv2
import numpy as np

folder = 'real_life_filter/'
filename = '01_original.png'
img = cv2.imread(folder+filename)


# from_x = 100
# to_x = 200
# from_y = 100
# to_y = 200
# section = img[from_x:to_x, from_y:to_y]
# img_name = '01_section'
# cv2.imwrite(folder+img_name+'.png', section)

# Denoising
denoised = cv2.fastNlMeansDenoisingColored(img,None,15,15,7,21)



img_name = '02_raised_contrast_04'
cv2.imwrite(folder+img_name+'.png', denoised)
