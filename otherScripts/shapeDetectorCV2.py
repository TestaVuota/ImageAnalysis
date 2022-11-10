#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 21:27:23 2021

@author: nicol
"""

import cv2, os
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from utils import dirpath, stackImages_SRC, stackImages, getContours_SRC
from  visualisation import (
    A, shift, Zsize, ZsizeL, nanostructuesArray, cropCoordinates,
    scharr,
    scharr_sobel,
    sobel,sobel_x,
    marr_hildreth,
    prewitt,
    Filters,
    edge_f,
    blur_filter1,
    blur_filter2,
    L,
    edge_detection,
    res_array, im,
)

# imNewA=np.array([res_array.T,res_array.T,res_array.T]).T
# imNew_array=(imNewA*((2**8-1)/(2**16-1))).astype(np.uint8)
# imNew=Image.fromarray(imNew_array)

# img = imNew_array

img = cv2.imread(r'C:\Users\nicol\Documents\Python\Detection_rectangle\820.0p5e-06.tiff')
imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
# getContours_SRC(imgCanny)

# imgBlank = np.zeros_like(img)
# imgStack = stackImages(0.8, ([img, imgGray, imgBlur],
#                              [imgCanny, imgContour, imgBlank]))

# cv2.imshow("Stack", imgStack)
cv2.imshow("imgContour", imgContour)
cv2.imshow("imgBlur", imgBlur)
cv2.imshow("imgCanny", imgCanny)
cv2.waitKey(0)
# plt.imshow("imgCanny", imgCanny)
# %%
newArray = (res_array*((2**8-1)/(2**16-1))).astype(np.uint8)
try:
    img_ = cv2.imdecode(newArray, cv2.IMREAD_GRAYSCALE)
except:
    pass
try:
    cv2.imshow("newArray", newArray)
    cv2.waitKey(0)
except:
    pass
# %%
# https://www.geeksforgeeks.org/python-detect-corner-of-an-image-using-opencv/
#-----------------------------------------------------------------------#
# https://www.programcreek.com/python/example/70414/cv2.IMREAD_GRAYSCALE
# img = np.fromstring(img.getvalue(), dtype='uint8')
# img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
# https://pythonexamples.org/python-opencv-read-image-cv2-imread/
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
# im = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
#-----------------------------------------------------------------------#

#-----------------------------------------------------------------------#
# im_template = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
#-----------------------------------------------------------------------#

# # Read in and make greyscale
# PILim = Image.open('image.jpg').convert('L')

# # Make Numpy/OpenCV-compatible version
# openCVim = np.array(PILim)
# PILim = Image.fromarray(openCVim)

# iterable = (x*x for x in range(5))
# np.fromiter(iterable, float)

# np.fromstring('1 2', dtype=int, sep=' ')
# array([1, 2])

#-----------------------------------------------------------------------#
# corner finder with cv2
# https://www.geeksforgeeks.org/python-detect-corner-of-an-image-using-opencv/
#-----------------------------------------------------------------------#


#-----------------------------------------------------------------------#
# https://pyimagesearch.com/2021/01/20/opencv-load-image-cv2-imread/
#-----------------------------------------------------------------------#
# %%
