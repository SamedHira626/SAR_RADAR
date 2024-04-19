#%% IMPORTS

import cv2
import numpy as np

from sarFilters import applyPipeline, median_filter, lee_filter, bilateral_filter
from morphologicalFunctions import applyClosing
from plotFunctions import plot_images

if 1:
    image = sar_image = cv2.imread('media/small.png', cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread('media/small1.png', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('media/small2.png', cv2.IMREAD_GRAYSCALE)
    cv2.circle(image2, (95,240), 10, (60,60,60), cv2.FILLED)
else:
    image = sar_image = cv2.imread('media/vhf1.jpg', cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread('media/vhf1.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('media/vhf2.jpg', cv2.IMREAD_GRAYSCALE)



#%% TESTING PIPELINE for NOISE FILTERING ON IMAGES

applyPipeline(sar_image, "median", True)
#applyPipeline(sar_image, "kuan")
applyPipeline(sar_image, "lee")
#applyPipeline(sar_image, "frost")
#applyPipeline(sar_image, "gamma")
#applyPipeline(sar_image, "sigma")
applyPipeline(sar_image, "bilateral")

window_size = 3

median = median_filter(image1, window_size)
lee = lee_filter(image1, window_size)
lee2 = np.uint8(lee)
bilateral = bilateral_filter(image1, window_size)

images = [image1, median, lee, bilateral,lee2]
titles = ['orj','median_filter','lee_filter','bilateral_filter','lee2']    
plot_images(images, titles, len(images))

#image = cv2.cvtColor(sar_image, cv2.COLOR_GRAY2BGR)
#applyPipeline(image, "lce")

#%% CHANGE DETECTION TESTING - DIFFERENCING

kernel = np.ones((3,3), dtype = np.uint8)

# Calculate absolute difference between the two images
difference = cv2.absdiff(image, image2) #equals to np.abs(image.astype(np.int8) - image2.astype(np.int8))

# Threshold the difference image to get binary mask of changes
_, thresholded = cv2.threshold(difference, 180, 255, cv2.THRESH_BINARY)
#thresholded = applyClosing(thresholded, kernel)

#applyPipeline(difference, "median")
_,_,_,_, eroded = applyPipeline(image, "lee")
_,_,_,_, eroded_ = applyPipeline(image2, "lee")

difference2 = cv2.absdiff(eroded, eroded_)

img1Cropped = image[200:260,75:180]
img2Cropped = image2[200:260,75:180]

difference_ = difference[200:260,75:180]
difference2_ = difference2[200:260,75:180]

images = [image, image2,    img1Cropped, img2Cropped,     difference,     thresholded,      difference2,  difference_,   difference2_]
titles = ['image','image2','img1Cropped','img2Cropped',  'orjDifference','thresholdedDif', 'pipelinedDif','orjDifZoom','pipelinedDifZoom']

plot_images(images, titles, len(images))

#%% CHANGE DETECTION TESTING - Coherence CHANGE DETECTION

"""
This code calculates the coherence between two input images using the cross-correlation method. It then applies a smoothing window 
to the coherence values and thresholds them to detect changes in the scene. Finally, it displays the original images, the coherence map, 
and the detected changes. That also uses "cv2.TM_CCORR_NORMED: Normalized cross-correlation between the template and the image"
"""

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
titles = ['image1','image2','coherence change']

plot_images(images, titles, len(images))

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

plot_images(images, titles, len(images))

#%% CHANGE DETECTION TESTING - K-MEANS CLUSTERING BASED CHANGE DETECTION

def unsupervised_change_detection(img1, img2, num_clusters, threshold): #TODO try with first image and try to remove differences
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
        
    # Calculate absolute difference between pixel values of the two images
    diff = np.abs(image.astype(np.int8) - image2.astype(np.int8))
#    diff[diff <= 20] = 0

    # Create binary mask for changed pixels
    change_mask = np.uint8(label_img1 != label_img2) * 255
    change_mask2 = np.uint8((label_img1 != label_img2) & (diff > threshold))* 255
 
    
    return change_mask, change_mask2

#applyLeeFilter = True;
#if True == applyLeeFilter:
#    image = lee_filter(image, window_size=5)
#    image2 = lee_filter(image2, window_size=5)

# Perform unsupervised change detection
num_clusters = 2  # Assuming two classes: unchanged and changed
change_mask, change_mask2 = unsupervised_change_detection(image, image2, num_clusters, 20)

image1_crop = image[200:260,75:180]
image2_crop = image2[200:260,75:180]
change_crop = change_mask[200:260,75:180]
change_crop2 = change_mask2[200:260,75:180]
    
images = [image,   image2,     change_mask,     change_mask2,       image1_crop, image2_crop,   change_crop, change_crop2]
titles = ['image1','image2','k-means changes', 'k-means with 20', 'image1_crop','image2_crop','k changes diff','k changes diff 20']

plot_images(images, titles, len(images))

#%%

i = np.abs(image.astype(np.int8) - image2.astype(np.int8))
i[i <= 20] = 0