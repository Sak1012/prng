import numpy as np
import cv2

def calculate_histogram(image):
    # Calculate the histogram of the image (256 bins for grayscale)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()  # Convert histogram to 1D array
    return hist

def histogram_variance(histogram):
    G = 256  # Number of gray levels
    variance = 0
    # Calculate the variance as defined in the formula
    for i in range(G):
        for j in range(G):
            variance += 0.5 * (histogram[i] - histogram[j])**2
    
    variance /= G**2
    return variance

def calculate_variance_reduction(var_plaintext, var_ciphertext):
    # Calculate the variance reduction ratio
    return (var_plaintext - var_ciphertext) / var_plaintext

# Load a grayscale image (plaintext) and its encrypted version (ciphertext)
plaintext_image = cv2.imread('catpix.jpg', cv2.IMREAD_GRAYSCALE)
ciphertext_image = cv2.imread('encrypted_image.png', cv2.IMREAD_GRAYSCALE)

# Calculate histograms for both images
hist_plaintext = calculate_histogram(plaintext_image)
hist_ciphertext = calculate_histogram(ciphertext_image)

# Calculate histogram variance for both images
var_plaintext = histogram_variance(hist_plaintext)
var_ciphertext = histogram_variance(hist_ciphertext)

# Calculate variance reduction
variance_reduction = calculate_variance_reduction(var_plaintext, var_ciphertext)

# Print results
print(f"Histogram Variance (Plaintext): {var_plaintext}")
print(f"Histogram Variance (Ciphertext): {var_ciphertext}")
print(f"Variance Reduction: {variance_reduction*100}")
