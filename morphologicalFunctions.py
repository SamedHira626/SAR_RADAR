#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def applyThreshold(image, minV, maxV):
    _, thresh_img = cv2.threshold(image, thresh = minV, maxval = maxV, type = cv2.THRESH_BINARY)
    return thresh_img

def applyErode(image, kernel, iterationNum):
    eroded = cv2.erode(image, kernel, iterations = iterationNum)
    return eroded

def applyDilate(image, kernel, iterationNum):
    dilated = cv2.dilate(image, kernel, iterations = iterationNum)
    return dilated

def applyClosing(image, kernel):
    closed = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_CLOSE, kernel)
    return closed

def applyOpening(image, kernel):
    opened = cv2.morphologyEx(image.astype(np.float32), cv2.MORPH_OPEN, kernel)
    return opened

def applyCropping(image, x1, x2, y1, y2):   
    crop = image[x1:x2,y1:y2]
    return crop

"""
Erosion: In image processing, erosion is a morphological operation that works by "eroding away" the boundaries of regions of foreground pixels 
(typically white pixels) in an image. It involves moving a structuring element (a small matrix) over the image and replacing each pixel with 
the minimum pixel value within the neighborhood defined by the structuring element. Erosion is often used for tasks such as removing small objects 
or thinning the borders of regions.

Dilation: Dilation is the opposite of erosion. It involves moving a structuring element over the image and replacing each pixel with the maximum 
pixel value within the neighborhood defined by the structuring element. Dilation causes regions of foreground pixels to expand. It's commonly used 
for tasks like joining broken parts of an object or filling in small holes within regions.

Opening: Opening is a morphological operation that combines erosion followed by dilation. It's useful for removing noise from images while preserving 
the shape and size of the foreground objects. Opening is achieved by first applying erosion to the image to remove small details, followed by dilation 
to restore the remaining objects to their approximate original size.

Closing: Closing is the opposite of opening, consisting of dilation followed by erosion. It's useful for closing small gaps or breaks within regions 
of foreground pixels. Closing is achieved by first applying dilation to the image to fill in small gaps or breaks within objects, followed by erosion 
to restore the objects to their approximate original size while removing noise from the background.

In summary, erosion and dilation modify the shape of objects in an image by either shrinking or expanding them, while opening and closing are 
combinations of these operations used for specific image enhancement tasks.
"""