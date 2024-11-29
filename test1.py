import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_adjacent_pixel_pairs(image, axis):
    """
    Extract adjacent pixel pairs along a specific axis.
    Axis=1 for horizontal, Axis=0 for vertical.
    """
    if axis == 1:  # Horizontal
        x = image[:, :-1].flatten()
        y = image[:, 1:].flatten()
    elif axis == 0:  # Vertical
        x = image[:-1, :].flatten()
        y = image[1:, :].flatten()
    return x, y

# Load images
original_image_path = "catpix.jpg"  # Replace with your image path

original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
encrypted_image = cv2.imread("encrypted_image.png", cv2.IMREAD_UNCHANGED)

# Compute adjacent pixel pairs
original_horizontal_x, original_horizontal_y = get_adjacent_pixel_pairs(original_image, axis=1)
encrypted_horizontal_x, encrypted_horizontal_y = get_adjacent_pixel_pairs(encrypted_image, axis=1)

# Plot scatter plots
plt.figure(figsize=(10, 5))

# Scatter plot for original image
plt.subplot(1, 2, 1)
plt.scatter(original_horizontal_x, original_horizontal_y, s=1, color='purple')
plt.title("Original Image: Adjacent Pixels")
plt.xlabel("Pixel gray value on location (x,y)")
plt.ylabel("Pixel gray value on location (x+1, y)")

# Scatter plot for encrypted image
plt.subplot(1, 2, 2)
plt.scatter(encrypted_horizontal_x, encrypted_horizontal_y, s=1, color='purple')
plt.title("Encrypted Image: Adjacent Pixels")
plt.xlabel("Pixel gray value on location (x,y)")
plt.ylabel("Pixel gray value on location (x+1, y)")

plt.tight_layout()
plt.show()
