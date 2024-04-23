#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

from plotFunctions import plot_images

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
    
    if True: #TODO remove when float32    
        filtered_image = np.uint8(filtered_image)

    return filtered_image

def gamma_filter(image, window_size, gamma):
    # Normalize pixel values to the range [0, 1]
    normalized_image = image.astype('float32') / 255.0
    
    # Apply gamma correction
    gamma_corrected_image = np.power(normalized_image, gamma)
    
    # Denormalize the image to the original range [0, 255]
    gamma_corrected_image = (gamma_corrected_image * 255).astype('uint8')
    
    return gamma_corrected_image

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

    filtered_image = cv2.medianBlur(image, ksize = window_size)
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

def bilateral_filter(image, window_size):
    
    window_size = 3 #TODO remove when testing is done
    
    # define parameters for bilateral filter
    diameter = window_size * window_size  # Diameter of each pixel neighborhood
    sigma_color = 50  # Filter sigma in the color space
    sigma_space = 75  # Filter sigma in the coordinate space
    
    # apply bilateral filtering
    filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return filtered_image

def kalman_filter(image):

    image = image.astype(np.float32)
    
    # Initialize Kalman filter parameters
    kalman = cv2.KalmanFilter(2, 1)  # 2 state variables (mean and variance), 1 measurement variable
    kalman.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)  # State transition matrix
    kalman.measurementMatrix = np.array([[1, 0]], dtype=np.float32)  # Measurement matrix
    kalman.processNoiseCov = np.array([[0.1, 0], [0, 0.1]], dtype=np.float32)  # Process noise covariance
    kalman.measurementNoiseCov = np.array([[10]], dtype=np.float32)  # Measurement noise covariance
    
    # Initialize state variables (mean and variance)
    state = np.zeros((2, 1), dtype=np.float32)
    kalman.statePost = state
    
    # Initialize filtered image
    filtered_image = np.zeros_like(image)
    
    # Iterate over each pixel in the image
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Predict next state
            predicted_state = kalman.predict()
            
            # Update measurement (current pixel intensity)
            measurement = np.array([[image[y, x]]], dtype=np.float32)
            
            # Update Kalman filter with measurement
            kalman.correct(measurement)
            
            # Retrieve filtered pixel value
            filtered_image[y, x] = kalman.statePost[0]
    
    return filtered_image.astype(np.uint8)

def applyPipeline(sar_image, filterType, draw = False):
       
    kernel = np.ones((3,3), dtype = np.uint8)

    _, thresh_img = cv2.threshold(sar_image, thresh = 180, maxval = 255, type = cv2.THRESH_BINARY)
    
    filter_functions = {
        "median": median_filter,
        "gauss": gauss_filter,
        "lee": lee_filter,
        "gamma": lambda img, window_size= 5, gamma=2.0: gamma_filter(img, window_size, gamma),
        "frost": lambda img, window_size= 5, alpha=1.5: frost_filter(img, window_size, alpha),
        "kuan": kuan_filter,
        "sigma": sigma_filter,
        "bilateral": bilateral_filter,
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
    
    if draw == True:
        images = [sar_image, thresh_img,  filtered,          filteredThresh,              threshClosing,                  eroded1,                          eroded2]
        titles = ['Original','thresh', filterType+'Filter', 'thres+'+filterType, 't+'+filterType[0]+'+closing','t+'+filterType[0]+'+closing+erode(1)','t+'+filterType[0]+'+closing+erode(2)']
            
        plot_images(images, titles, 7)  
    
    return filtered, filteredThresh, threshClosing, eroded1, eroded2
