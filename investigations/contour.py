from __future__ import print_function
import cv2
import numpy as np
import argparse
import random as rng



im = cv2.imread("image_above.jpg")
h,w,chn = im.shape
seed = (w/2,h/2)

mask = np.zeros((h+2,w+2),np.uint8)

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)

num,im,mask,rect = cv2.floodFill(im, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)

cv2.imwrite("flood_fill.png", mask)
