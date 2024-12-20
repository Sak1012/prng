import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
import math
from skimage.transform import resize
# Flask app endpoints
GENERATE_KEY_URL = "http://localhost:5000/generate-key"
ENCRYPT_URL = "http://localhost:5000/encrypt-image"
DECRYPT_URL = "http://localhost:5000/decrypt-image"

def encrypt_image_via_flask(image_path, key):
    """
    Encrypts an image using a Flask app.
    Parameters:
    - image_path: Path to the plain image file.
    - key: Key for encryption.

    Returns:
    - Encrypted image as a NumPy array.
    - IV used for encryption.
    - Width and height of encrypted image
    """
    with open(image_path, "rb") as image_file:
        response = requests.post(ENCRYPT_URL, files={"image": image_file}, data={"key": key})
        response.raise_for_status()
        
        iv = response.headers.get("X-Iv")
        width = int(response.headers.get("X-Width"))
        height = int(response.headers.get("X-Height"))
        encrypted_image = np.frombuffer(response.content, dtype=np.uint8)
        return encrypted_image, iv, width, height


def calculate_npcr_uaci_resized(ciphertext1, ciphertext2, width1, height1, width2, height2):
    """
    Calculate NPCR and UACI between two ciphertext images, cropping them to the same dimensions.
    
    Parameters:
    - ciphertext1: NumPy array of the first ciphertext image.
    - ciphertext2: NumPy array of the second ciphertext image.
    - width1, height1: Dimensions of the first ciphertext image.
    - width2, height2: Dimensions of the second ciphertext image.
    
    Returns:
    - NPCR: Number of Pixel Change Rate as a percentage.
    - UACI: Unified Average Change Intensity as a percentage.
    """
    print(f"Ciphertext1 dimensions: {width1}x{height1}")
    print(f"Ciphertext2 dimensions: {width2}x{height2}")

    # Ensure array sizes match declared dimensions
    def reshape_and_validate(array, current_width, current_height):
        expected_size = current_width * current_height
        actual_size = array.size
        
        if actual_size < expected_size:
            # Pad with zeros if too small
            array = np.pad(array, (0, expected_size - actual_size), mode='constant')
        elif actual_size > expected_size:
            # Trim if too large
            array = array[:expected_size]
        
        # Reshape to expected dimensions
        return array.reshape(current_height, current_width)

    ciphertext1 = reshape_and_validate(ciphertext1, width1, height1)
    ciphertext2 = reshape_and_validate(ciphertext2, width2, height2)

    # Determine the minimum dimensions for cropping
    min_width = min(width1, width2)
    min_height = min(height1, height2)

    # Crop both ciphertexts to the minimum dimensions (central crop)
    def crop_to_target(array, target_width, target_height):
        start_x = (array.shape[1] - target_width) // 2
        start_y = (array.shape[0] - target_height) // 2
        cropped_array = array[start_y:start_y+target_height, start_x:start_x+target_width]
        return cropped_array

    ciphertext1_cropped = crop_to_target(ciphertext1, min_width, min_height)
    ciphertext2_cropped = crop_to_target(ciphertext2, min_width, min_height)

    # NPCR calculation
    difference_matrix = (ciphertext1_cropped != ciphertext2_cropped).astype(int)
    total_pixels = min_width * min_height
    npcr = np.sum(difference_matrix) / total_pixels * 100
    
    # UACI calculation
    intensity_difference = np.abs(ciphertext1_cropped - ciphertext2_cropped)
    uaci = np.sum(intensity_difference) / (total_pixels * 255) * 100
    
    return npcr, uaci, ciphertext1_cropped, ciphertext2_cropped

if __name__ == "__main__":
    # Step 1: Load the plain image
    image_path = "colors.jpg"
    
    # Load and convert to grayscale
    plain_image = Image.open(image_path).convert("L")
    plain_array = np.array(plain_image)
    
    print(f"Original image shape: {plain_array.shape}")
    
    # Step 2: Use the provided key
    key = "748d2185478afc7fd12618b769a937bd"
    print(f"Encryption Key: {key}")
    
    # Step 3: Encrypt the plain image
    print("Encrypting original image...")
    ciphertext1, iv1, width1, height1 = encrypt_image_via_flask(image_path, key)
    print(f"Ciphertext1 dimensions: {width1}x{height1}")
    
    # Step 4: Modify the plain image
    modified_plain_array = plain_array.copy()
    modified_plain_array[0, 0] = np.uint8((int(modified_plain_array[0, 0]) + 1) % 256)
    modified_image = Image.fromarray(modified_plain_array)
    modified_image_path = "modified_catpix.png"
    modified_image.save(modified_image_path)
    
    # Step 5: Encrypt the modified image
    print("Encrypting modified image...")
    ciphertext2, iv2, width2, height2 = encrypt_image_via_flask(modified_image_path, key)
    print(f"Ciphertext2 dimensions: {width2}x{height2}")
    
    # Step 6: Calculate NPCR and UACI
    print("\nCalculating NPCR and UACI...")
    npcr, uaci, c1_reshaped, c2_reshaped = calculate_npcr_uaci_resized(
        ciphertext1, ciphertext2, width1, height1, width2, height2
    )
    print(f"NPCR: {npcr:.2f}%")
    print(f"UACI: {uaci:.2f}%")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(plain_array, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Modified image
    plt.subplot(1, 3, 2)
    plt.imshow(modified_plain_array, cmap='gray')
    plt.title("Modified Image\n(1 pixel changed)")
    plt.axis("off")
    
    # Difference map
    plt.subplot(1, 3, 3)
    difference_image = np.abs(c1_reshaped - c2_reshaped)
    plt.imshow(difference_image, cmap="hot")
    plt.colorbar()
    plt.title("Ciphertext Differences")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()