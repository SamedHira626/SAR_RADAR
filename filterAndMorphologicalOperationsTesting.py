#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#IMPORTS

import cv2
import numpy as np
import matplotlib.pyplot as plt

from morphologicalFunctions import applyErode, applyDilate, applyOpening, applyClosing, applyThreshold, applyCropping
from sarFilters import lee_filter, gamma_filter, kuan_filter, median_filter, sigma_filter, local_contrast_enhancement, frost_filter, bilateral_filter
from plotFunctions import plot_images

sar_image = cv2.imread('media/small.png', cv2.IMREAD_GRAYSCALE)

#%% MORPHOLOGICAL OPERATIONS - 1

kernel = np.ones((3,3), dtype = np.uint8)

thresh_img = applyThreshold(sar_image, 150, 255) 

eroded = applyErode(thresh_img, kernel, 1)

dilated = applyDilate(thresh_img, kernel, 1)

opening = applyOpening(thresh_img, kernel)

closing = applyClosing(thresh_img, kernel)

images = [sar_image, thresh_img, eroded, dilated, opening, closing]
titles = ['Original Image','Threshold SAR Image','thresh+eroded',
          'thresh+dilated','thresh+opening','thresh+closing']
    
plot_images(images, titles, 6)  

#%% MORPHOLOGICAL OPERATIONS - 2

kernel = np.ones((3,3), dtype = np.uint8)

_, thresh_img = cv2.threshold(sar_image, thresh = 180, maxval = 255, type = cv2.THRESH_BINARY) 
thresh_img2 = cv2.adaptiveThreshold(sar_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

dilated = cv2.dilate(thresh_img, kernel, iterations = 1)

threshClosing = cv2.morphologyEx(thresh_img.astype(np.float32), cv2.MORPH_CLOSE, kernel)
thresh2closing = cv2.morphologyEx(threshClosing.astype(np.float32), cv2.MORPH_CLOSE, kernel)

thresh2closingOpening = cv2.morphologyEx(threshClosing.astype(np.float32), cv2.MORPH_OPEN, kernel) 

eroded1 = cv2.erode(threshClosing, kernel, iterations = 1)
eroded2 = cv2.erode(threshClosing, kernel, iterations = 2)

images = [sar_image, thresh_img, threshClosing, eroded1, eroded2, thresh2closingOpening]
titles = ['Original Image','Threshold SAR Image','thresh+closing',
          'thresh+closing+erode(1)','thresh+closing+erode(2)','gamma']
    
plot_images(images, titles, 6)  

#%% FILTER OPERATIONS - 2

_, thresh_img = cv2.threshold(sar_image, thresh = 180, maxval = 255, type = cv2.THRESH_BINARY) 
lee = lee_filter(sar_image, window_size=5)

gamma = gamma_filter(sar_image, window_size=3, gamma=2.0)

frost = frost_filter(sar_image, window_size=3, alpha=1.5)
#frost2 = frost_filter2(sar_image, window_size=3, alpha=1.5)

kuan = kuan_filter(sar_image, window_size=5)

bilateral = bilateral_filter(sar_image, window_size=5)

median = median_filter(sar_image, window_size=5)
medianThresh = median_filter(thresh_img, window_size=5)

sigma = sigma_filter(sar_image, window_size = 5)

image = cv2.cvtColor(sar_image, cv2.COLOR_GRAY2BGR)
lce = local_contrast_enhancement(image, neighborhood_size=15, clip_limit=2.0) #needs 3 channel image

images = [sar_image, thresh_img,  lee,   gamma,       frost, kuan,    median,    bilateral,   lce]
titles = ['Original','Threshold','lee', 'gamma 2.0', 'frost', 'kuan', 'median', 'bilateral', 'lce']

x1, x2, y1, y2 = [75, 180, 200, 260]
images = [applyCropping(image, x1, x2, y1, y2) for image in images] #to crop images

plot_images(images, titles, len(images))
