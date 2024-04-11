#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from plotFunctions import *

def lee_filter(image, window_size):
    # Convert image to float32
    image = np.float32(image)

    # Calculate local mean using a rectangular window
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))

    # Calculate local variance using a rectangular window
    mean_of_square = cv2.boxFilter(image * image, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_of_square - mean * mean

    # Compute the ratio between local variance and global variance
    global_variance = np.var(image)
    ratio = np.minimum(variance / (variance + global_variance), 1.0)

    # Calculate the filtered image using Lee filter
    filtered_image = mean + ratio * (image - mean)

    return filtered_image

def gamma_filter(image, window_size, gamma):
    # Convert the input image to floating-point representation
    image_float = np.float32(image)

    # Define the filter window
    filter_window = np.ones((window_size, window_size))

    # Apply gamma correction to the image
    gamma_corrected = cv2.pow(image_float, gamma)

    # Apply filter to the gamma-corrected image
    filtered_image = cv2.filter2D(gamma_corrected, -1, filter_window)

    # Convert the filtered image back to the original data type
    filtered_image = np.uint8(filtered_image)

    return filtered_image

def frost_filter(image, window_size, alpha):
    # Convert image to float32
    image = np.float32(image)

    # Calculate local statistics using a rectangular window
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    mean_of_square = cv2.boxFilter(image * image, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_of_square - mean * mean

    # Compute the filtered image using Frost filter
    filtered_image = image - alpha * variance

    return filtered_image

def frost_filter2(image, window_size=3, alpha=1.5):
    # Calculate local means
    local_means = cv2.blur(image, (window_size, window_size))
    
    # Calculate local variances
    local_variances = cv2.blur(image**2, (window_size, window_size)) - local_means**2
    
    # Estimate noise variance
    noise_variance = np.mean(local_variances)
    
    # Calculate filtering parameter
    beta = alpha * noise_variance
    
    # Apply Frost filter
    filtered_image = image * (1 - beta / (local_variances + beta))
    
    return filtered_image

def kuan_filter(image, window_size):
    # Convert image to float32
    image = np.float32(image)

    # Calculate local statistics using a rectangular window
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    mean_of_square = cv2.boxFilter(image * image, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_of_square - mean * mean

    # Compute the filtered image using Kuan filter
    filtered_image = mean + (image - mean) * (variance / (variance + mean**2))

    return filtered_image

def median_filter(image, window_size):

    filtered_image = cv2.medianBlur(image, window_size)
    return filtered_image

def gauss_filter(image, window_size):
    
    filtered_image = cv2.GaussianBlur(image, ksize = (window_size,window_size), sigmaX = 7)
    return filtered_image

def sigma_filter(image, window_size):
 
    # Calculate local statistics using a rectangular window
    mean = cv2.boxFilter(image, ddepth=-1, ksize=(window_size, window_size))
    mean_of_square = cv2.boxFilter(image * image, ddepth=-1, ksize=(window_size, window_size))
    variance = mean_of_square - mean * mean

    # Compute the filtered image using Sigma filter
    filtered_image = mean + variance / (mean**2 + 1e-6)

    return filtered_image

def local_contrast_enhancement(image, neighborhood_size=15, clip_limit=2.0):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split LAB image into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(neighborhood_size, neighborhood_size))
    enhanced_l = clahe.apply(l)
    
    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab = cv2.merge((enhanced_l, a, b))
    
    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def applyPipeline(sar_image, filterType):
       
    kernel = np.ones((3,3), dtype = np.uint8)

    _, thresh_img = cv2.threshold(sar_image, thresh = 104, maxval = 255, type = cv2.THRESH_BINARY)
    
    filter_functions = {
        "median": median_filter,
        "gauss": gauss_filter,
        "lee": lee_filter,
        "gamma": lambda img, window_size= 5, gamma=1.5: gamma_filter(img, window_size, gamma),
        "frost": lambda img, window_size= 5, alpha=1.5: frost_filter(img, window_size, alpha),
        "kuan": kuan_filter,
        "sigma": sigma_filter,
        "lce" : lambda img, neighborhood_size=15, clip_limit=2.0: local_contrast_enhancement(img, neighborhood_size=15, clip_limit=2.0)
    }
    
    filter_function = filter_functions[filterType]
    
    if filterType == "lce":      
        filtered = filter_function(sar_image, neighborhood_size=15, clip_limit=2.0)
        filteredThresh = filter_function(thresh_img, neighborhood_size=15, clip_limit=2.0)
    
    else:
        filtered = filter_function(sar_image, window_size=5)
        filteredThresh = filter_function(thresh_img, window_size=5)
    
    threshClosing = cv2.morphologyEx(filteredThresh.astype(np.float32), cv2.MORPH_CLOSE, kernel)
 
    eroded1 = cv2.erode(threshClosing, kernel, iterations = 1)
    eroded2 = cv2.erode(threshClosing, kernel, iterations = 2)
    
    images = [sar_image, thresh_img,  filtered,          filteredThresh,              threshClosing,                  eroded1,                          eroded2]
    titles = ['Original','thresh', filterType+'Filter', 'thres+'+filterType, 't+'+filterType[0]+'+closing','t+'+filterType[0]+'+closing+erode(1)','t+'+filterType[0]+'+closing+erode(2)']
        
    plot_images(images, titles, 7)  