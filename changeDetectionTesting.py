#%% IMPORTS

import cv2
import numpy as np

from sarFilters import applyPipeline
from plotFunctions import plot_images

image = sar_image = cv2.imread('media/small.png', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('media/small1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('media/small2.png', cv2.IMREAD_GRAYSCALE)

cv2.circle(image2, (95,240), 10, (60,60,60), cv2.FILLED)

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
titles = ['image','image2','crop1','crop2','orjDifference', 'thresholdedDif', 'pipelinedDif','orjDifZoom','pipelinedDifZoom']

plot_images(images, titles, 9)
#%% CHANGE DETECTION TESTING - Coherence CHANGE DETECTION

def compute_coherence(img1_data, img2_data):
    
    img1 = img1_data.astype(np.complex64)
    img2 = img2_data.astype(np.complex64)

    # compute absolute values
    abs_img1 = np.abs(img1)
    abs_img2 = np.abs(img2)
       
    # avoid division by zero
    mask = (abs_img1 * abs_img2 == 0)
    abs_img1[mask] = 1  # Set zero values to 1 to avoid division by zero
    abs_img2[mask] = 1

    # compute coherence
    coherence = np.abs(np.conj(img1) * img2) / (np.abs(img1) * np.abs(img2))
    
    return coherence
    
def detect_changes(coherence, threshold=0.8):
    # handle invalid values
    coherence[np.isnan(coherence)] = 0
    coherence[np.isinf(coherence)] = 0
    
    change_mask = (coherence < threshold).astype(np.uint8) * 255
    
    return change_mask

coherence = compute_coherence(image1, image2)

if coherence is not None:
    change_mask = detect_changes(coherence)
else:
    print("Coherence computation failed. Check input images and paths.")

images = [image1, image2, change_mask]
titles = ['image','image2','change']

plot_images(images, titles, 3)

#%% CHANGE DETECTION TESTING - INTENSITY BASED CHANGE DETECTION

def intensity_change_detection(image1, image2, threshold):

    # Calculate absolute difference between the two images
    difference = cv2.absdiff(image1, image2)

    # Apply threshold to detect significant intensity changes
    _, binary = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    return binary

# Set intensity change threshold
threshold = 180

# Perform intensity-based change detection
changes = intensity_change_detection(image1, image2, threshold)

images = [image1, image2, changes]
titles = ['image1','image2','Intensity-based Changes']

plot_images(images, titles, 3)

#%% CHANGE DETECTION TESTING - K-MEANS CLUSTERING BASED CHANGE DETECTION

def unsupervised_change_detection(img1, img2, num_clusters):
    """
    Perform unsupervised change detection using k-means clustering
    """
    assert img1.shape == img2.shape, "Images must have the same size"
    
    # Stack images vertically for clustering
    stacked_img = np.hstack((img1, img2))
    
    # Reshape stacked image for k-means clustering
    stacked_img_flat = stacked_img.reshape((-1, 1))

    # Convert to float32
    stacked_img_flat = np.float32(stacked_img_flat)
    
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, label, center = cv2.kmeans(stacked_img_flat, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Reshape labels to match the shape of the stacked image
    label = label.reshape(stacked_img.shape[:2])
    
    # Split label into two separate images
    height, width = img1.shape[:2]
    label_img1 = label[:, :width]
    label_img2 = label[:, width:]
    
    # Create binary mask for changed pixels
    change_mask = np.uint8(label_img1 != label_img2) * 255
    
    return change_mask

# Perform unsupervised change detection
num_clusters = 2  # Assuming two classes: unchanged and changed
change_mask = unsupervised_change_detection(image1, image2, num_clusters)

images = [image1, image2, change_mask]
titles = ['image1','image2','Changes']

plot_images(images, titles, 3)
