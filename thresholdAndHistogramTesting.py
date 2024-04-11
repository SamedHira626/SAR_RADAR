#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# THRESHOLD TESTING

import cv2
import numpy as np
from plotFunctions import plot_images

img = sar_image = cv2.imread('media/small.png', cv2.IMREAD_GRAYSCALE)

ret,thresh1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,180,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,180,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,180,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,180,255,cv2.THRESH_TOZERO_INV)
ret,thresh6 = cv2.threshold(img,180,255,cv2.THRESH_OTSU)
ret,thresh7 = cv2.threshold(img,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
adaptiveThresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','OTSU','BINARY+OTSU','ADAPTIVE']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh6, thresh7, adaptiveThresh]

plot_images(images, titles, 8)

#%% HISTOGRAM TESTING

def sar_histogram(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Calculate the histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    return hist

def otsu_threshold(hist):
    
    total_pixels = np.sum(hist)

    probabilities = hist / total_pixels
    max_variance = 0
    threshold = 0

    # Calculate optimal threshold using Otsu's method
    for t in range(1, 256):
        # Background class probabilities and weights
        wb = np.sum(probabilities[:t])
        mb = np.sum(probabilities[:t] * np.arange(t)) / wb

        # Foreground class probabilities and weights
        wf = np.sum(probabilities[t:])
        mf = np.sum(probabilities[t:] * np.arange(t, 256)) / wf

        # Calculate between-class variance
        variance = wb * wf * (mb - mf) ** 2

        # Update threshold if variance is maximum
        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold

# Calculate the histogram
histogram = sar_histogram(sar_image)

# Apply Otsu's thresholding algorithm
threshold_value = otsu_threshold(histogram)

# Apply thresholding
_, thresholded_image = cv2.threshold(sar_image, threshold_value, 255, cv2.THRESH_BINARY)

titles = ['Original Image','Thresholded Image (Otsu)']
images = [img, thresh1]

plot_images(images, titles, 2)

