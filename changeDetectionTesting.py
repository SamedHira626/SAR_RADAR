#%% IMPORTS

import cv2
import numpy as np
import matplotlib.pyplot as plt

from sarFilters import applyPipeline
from plotFunctions import plot_images

image = sar_image = cv2.imread('media/small.png', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('media/small1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('media/small2.png', cv2.IMREAD_GRAYSCALE)

#%% TESTING PIPELINE for NOISE FILTERING ON IMAGES

applyPipeline(sar_image, "median", True)
#applyPipeline(sar_image, "kuan")
applyPipeline(sar_image, "lee", True)
#applyPipeline(sar_image, "frost")
#applyPipeline(sar_image, "gamma")
#applyPipeline(sar_image, "sigma")

#image = cv2.cvtColor(sar_image, cv2.COLOR_GRAY2BGR)
#applyPipeline(image, "lce")

#%% CHANGE DETECTION TESTING - DIFFERENCING

cv2.circle(image2, (100,150), 20, (50,50,50), cv2.FILLED)

# Calculate absolute difference between the two images
difference = cv2.absdiff(image1, image2)

# Threshold the difference image to get binary mask of changes
_, thresholded = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

#applyPipeline(difference, "median")
_,_,_,_, eroded = applyPipeline(image1, "lee")
_,_,_,_, eroded_ = applyPipeline(image2, "lee")

difference2 = cv2.absdiff(eroded, eroded_)

img1Cropped = image1[200:260,75:180]
img2Cropped = image2[200:260,75:180]

difference_ = difference[200:260,75:180]
difference2_ = difference2[200:260,75:180]

images = [image1, image2, img1Cropped, img2Cropped, difference, thresholded, difference2, difference_, difference2_]
titles = ['image','image2','crop1','crop2','orjDifference', 'thresholdedDif', 'pipelinedDif','orjDif1','pipelinedDif2']

plot_images(images, titles, 9)

#%% CHANGE DETECTION TESTING - Coherence CHANGE DETECTION

def calculate_coherence(image1, image2, window_size=3):
    
    # Calculate cross-correlation between the two images
    correlation = cv2.matchTemplate(image1, image2, cv2.TM_CCORR_NORMED)
    
    # Calculate coherence
    coherence = np.abs(correlation)
    
    # Apply a window to smooth the coherence values
    kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    coherence = cv2.filter2D(coherence, -1, kernel)
    
    return coherence

def detect_changes(coherence, threshold):
    # Threshold the coherence to detect changes
    changes = coherence < threshold
    
    return changes

# Load two images
image1 = cv2.imread('media/small1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('media/small2.png', cv2.IMREAD_GRAYSCALE)

# Calculate coherence
coherence = calculate_coherence(image1, image2)

# Set a coherence threshold
threshold = 0.5

# Detect changes using coherence thresholding
changes = detect_changes(coherence, threshold)
changes = np.uint8(changes) * 255

images = [image1, image2, changes]
titles = ['image1','image2','Changes']

plot_images(images, titles, 3)

#%% CHANGE DETECTION TESTING - INTENSITY BASED CHANGE DETECTION

def intensity_change_detection(image1, image2, threshold):

    # Calculate absolute difference between the two images
    difference = cv2.absdiff(image1, image2)

    # Apply threshold to detect significant intensity changes
    _, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    return binary



# Set intensity change threshold
threshold = 30

# Perform intensity-based change detection
changes = intensity_change_detection(image1, image2, threshold)

images = [image1, image2, changes]
titles = ['image1','image2','Intensity-based Changes']

plot_images(images, titles, 3)
