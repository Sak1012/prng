import os
import io
import time
import ast
import cv2
import numpy as np
import matplotlib.pyplot as plt
import psutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PIL import Image, ImageChops
from scipy.integrate import solve_ivp
from scipy.stats import chisquare, pearsonr, entropy
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.transform import resize
import requests
from io import BytesIO
app = Flask(__name__)
CORS(app, expose_headers=["X-Encryption-Key", "X-Data-Length", "X-Iv"])
class PrngOscillator:
    def __init__(self):
        self.params = {
            "g1": 1925.0, "gkv": 1700.0, "gL": 7.0, "v1": 100.0,
            "vk": -75.0, "vL": -40.0, "vc": 100.0, "kc": 3.3 / 18.0,
            "rao": 0.27, "lamn": 230.0, "gkc": 12.0, "k0": 0.1,
            "k1": 1.0, "k2": 0.5, "E": 0.0,
            "ohm1": 1.414, "ohm2": 0.618,
        }
        entropy = os.urandom(32)
        self.initial_state = [(b % 100) / 100.0 for b in entropy[:4]]

    def system_equations(self, t, v):
        x1, y1, z1, w1 = v
        p = self.params
        phi_ext = p["E"] * (np.sin(p["ohm1"] * t) + np.sin(p["ohm2"] * t))
        mp = np.tanh(w1)

        tau_n, n_inf, h_inf, m_inf = 1, 0.5, 0.5, 0.5
        dx1 = (
            p["g1"] * m_inf**3 * h_inf * (p["v1"] - x1)
            + p["gkv"] * y1**4 * (p["vk"] - x1)
            + p["gkc"] * (z1 / (1 + z1)) * (p["vk"] - x1)
            + p["gL"] * (p["vL"] - x1)
            - p["k0"] * mp * x1
        )
        dy1 = (n_inf - y1) / tau_n
        dz1 = (m_inf**3 * h_inf * (p["vc"] - x1) - p["kc"] * z1) * p["rao"]
        dw1 = p["k1"] * x1 - p["k2"] * w1 + phi_ext
        return [dx1, dy1, dz1, dw1]

    def simulate(self, t_span, dt=0.01, transient_time=1000.0):
        t_eval = np.arange(t_span[0], t_span[1], dt)
        solution = solve_ivp(
            self.system_equations,
            t_span,
            self.initial_state,
            method="RK45",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-9,
        )
        transient_points = int(transient_time / dt)
        return solution.t[transient_points:], solution.y[:, transient_points:]

    def generate_prng_bits(self, trajectory, num_bits=128):
        bits = [1 if trajectory[0][i % len(trajectory[0])] > 0 else 0 for i in range(num_bits)]
        return np.array(bits)

    def lfsr_post_processing(self, bits, taps=(0, 2, 3, 5)):
        lfsr_bits = []
        state = bits[:8].tolist()
        for bit in bits:
            feedback = sum(state[t] for t in taps) % 2
            lfsr_bits.append(state.pop(0))
            state.append(feedback)
        return np.array(lfsr_bits)
def encrypt_image(image_bytes, key):
    img = Image.open(io.BytesIO(image_bytes))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=img.format or "PNG")
    image_bytes = img_buffer.getvalue()
    img_format = img.format or "PNG"
    format_bytes = img_format.encode("utf-8")
    data_to_encrypt = len(format_bytes).to_bytes(1, "big") + format_bytes + image_bytes
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data_to_encrypt, AES.block_size)
    encrypted_data = cipher.encrypt(padded_data)
    return encrypted_data,iv

def decrypt_image(encrypted_data, key,iv):
    ciphertext = encrypted_data
    iv = ast.literal_eval(iv)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    decrypted_data = unpad(decrypted_padded, AES.block_size)
    format_length = decrypted_data[0]
    img_format = decrypted_data[1 : format_length + 1].decode("utf-8")
    
    image_data = decrypted_data[format_length + 1 :]

    return Image.open(io.BytesIO(image_data)), img_format

def compare_images(img1, img2):
    try:
        response = {}
        response["size"] = f"Image sizes First: {img1.size}, Second: {img2.size}"
        original_pixels = list(img1.getdata())
        decrypted_pixels = list(img2.getdata())
        response["pixel count"] =f"Pixel count .First : {len(original_pixels)},Second : {len(decrypted_pixels)}"
        mismatches = []
        for i, (orig_pixel, decrypt_pixel) in enumerate(zip(original_pixels, decrypted_pixels)):
            if any(abs(o - d) > 10 for o, d in zip(orig_pixel, decrypt_pixel)):
                mismatches.append((i, orig_pixel, decrypt_pixel))
        error_msg = f"Found {len(mismatches)} pixel mismatches. "
    except:
        error_msg = "Comparsion not allowed"
    response["mismatch"] = error_msg        
    return response

def test_randomness(bits):
    counts = np.bincount(bits, minlength=2)
    expected = np.full_like(counts, len(bits) / 2)
    chi2, p_value = chisquare(counts, expected)
    return {
        "bit_distribution": {"0s": int(counts[0]), "1s": int(counts[1])},
        "chi_square_statistic": float(chi2),
        "p_value": float(p_value)
    }

def log_resource_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "memory_usage_mb": mem_info.rss / 1024**2,
        "cpu_usage_percent": psutil.cpu_percent(interval=1)
    }
def calculate_correlation(image_data1, image_data2):
    return pearsonr(image_data1.flatten(), image_data2.flatten())[0]

def calculate_image_correlations(original_data, cipher_data):
    # Vertical correlation (across columns)
    vert_correlation = [calculate_correlation(original_data[:, i], cipher_data[:, i]) for i in range(original_data.shape[1])]
    vert_cor = np.mean(vert_correlation)

    # Horizontal correlation (across rows)
    hor_correlation = [calculate_correlation(original_data[i, :], cipher_data[i, :]) for i in range(original_data.shape[0])]
    hor_cor = np.mean(hor_correlation)

    # Diagonal correlation
    diag_correlation = [calculate_correlation(np.diagonal(original_data, offset=i), np.diagonal(cipher_data, offset=i)) for i in range(-original_data.shape[0] + 1, original_data.shape[1])]
    diag_cor = np.mean(diag_correlation)

    return vert_cor, hor_cor, diag_cor

mhho_prng = PrngOscillator()
def generate_encryption_key():
    try:
        # Simulate to get initial random bits
        _, trajectory = mhho_prng.simulate((0, 2000))
        random_bits = mhho_prng.generate_prng_bits(trajectory)
        lfsr_bits = mhho_prng.lfsr_post_processing(random_bits)
        
        entropy = os.urandom(16)
        entropy_bits = np.frombuffer(entropy, dtype=np.uint8).reshape(-1) % 2
        extended_entropy_bits = np.tile(entropy_bits, 8)[:128]
        
        mixed_bits = (lfsr_bits + extended_entropy_bits) % 2
        
        key = bytes(
            int("".join(map(str, mixed_bits[i : i + 8])), 2) for i in range(0, 128, 8)
        )
        
        return {
            "key": key.hex(),
            "randomness_test": test_randomness(mixed_bits)
        }
    except Exception as e:
        return {"error": str(e)}
@app.route("/generate-key")
def key_gen():
    key_resp = generate_encryption_key()
    key_hex =key_resp['key']  
    key = bytes.fromhex(key_hex)
    return jsonify({
        "key" : f"{key.hex()}"
    })

@app.route('/encrypt-image', methods=['POST'])
def encrypt_image_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        image_file = request.files['image']
        key = request.form["key"]  
        image_bytes = image_file.read() 
        hex_int = int(key, 16) 
        new_int = hex_int + 0x200
        key = hex(new_int)[2:] 
        key = bytes.fromhex(key)
        encrypted_data,iv = encrypt_image(image_bytes, key)
        data_length = len(encrypted_data)

        length_bytes = data_length.to_bytes(4, byteorder="big")
        combined_data = length_bytes + encrypted_data

        width = 1080
        height = (len(combined_data) + width - 1) // width
        encrypted_pixels = list(combined_data) + [0] * (width * height - len(combined_data))  # Pad with zeros
        encrypted_image = Image.frombytes("L", (width, height), bytes(encrypted_pixels))
        image_buffer = io.BytesIO()
        encrypted_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        response = send_file(
            image_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='encrypted_image.png'
        )
        print(key.hex())
        response.headers['X-Iv'] = iv
        response.headers["X-Data-Length"] = str(data_length)

        print(response.headers)

        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/decrypt-image', methods=['POST'])
def decrypt_image_endpoint():
    try:
        breakpoint()
        encrypted_file = request.files['file']
        key_hex = request.form['key']
        print(key_hex)
        iv = request.form['iv']
        hex_int = int(key, 16) 
        new_int = hex_int + 0x200
        key = hex(new_int)[2:] 
        key = bytes.fromhex(key_hex)
        encrypted_image = Image.open(encrypted_file)
        encrypted_pixels = encrypted_image.tobytes()
        length_bytes = encrypted_pixels[:4]
        data_length = int.from_bytes(length_bytes, byteorder="big")
        encrypted_data = encrypted_pixels[4:4 + data_length]
        decrypted_img, img_format = decrypt_image(encrypted_data, key,iv)
        img_buffer = io.BytesIO()
        decrypted_img.save(img_buffer, format=img_format)
        img_buffer.seek(0)

        return send_file(
            img_buffer,
            mimetype=f'image/{img_format.lower()}',
            as_attachment=True,
            download_name=f'decrypted_image.{img_format.lower()}'
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def calculate_correlation(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]

def calculate_directions_correlation(image):
    """Calculate horizontal, vertical, and diagonal correlations for an image."""
    # Convert to grayscale if needed
    if image.ndim == 3 and image.shape[2] == 4:  # Check for 4 channels (RGBA)
        image = image[:, :, :3]  # Drop the alpha channel
    if image.ndim == 3:  # RGB image
        image = rgb2gray(image)

    # Ensure image is 2D
    assert image.ndim == 2, "Image must be grayscale (2D)."
    
    # Shift operations to extract neighboring pixels
    horizontal = calculate_correlation(image[:, :-1], image[:, 1:])  # Right neighbor
    vertical = calculate_correlation(image[:-1, :], image[1:, :])    # Bottom neighbor
    diagonal = calculate_correlation(image[:-1, :-1], image[1:, 1:]) # Bottom-right neighbor

    return {
        "hcorr": round(horizontal, 5),
        "vcorr": round(vertical, 5),
        "dcorr": round(diagonal, 5)
    }

@app.route('/corr-calc', methods=['POST'])
def image_correlations():
    """
    POST Route to calculate horizontal, vertical, and diagonal correlations for two images.
    Input: 'original' and 'cipher' image files
    Output: JSON with correlation coefficients for both images
    """
    # Check if both images are provided
    if 'original' not in request.files or 'cipher' not in request.files:
        return jsonify({"error": "Both 'original' and 'cipher' images must be provided"}), 400

    # Load images
    original_file = request.files['original']
    cipher_file = request.files['cipher']
    
    try:
        original_image = imread(original_file)
        cipher_image = imread(cipher_file)
    except Exception as e:
        return jsonify({"error": f"Failed to read images: {str(e)}"}), 400

    try:
        # Calculate correlations for each image
        original_results = calculate_directions_correlation(original_image)
        cipher_results = calculate_directions_correlation(cipher_image)
        
        # Prepare and return the response
        response = {
                "original": original_results,
                "cipher": cipher_results
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Error during calculation: {str(e)}"}), 500

def compute_adjacent_pixel_difference(image, axis):
    """
    Compute the difference between adjacent pixels along the specified axis.
    Axis=1 for horizontal, Axis=0 for vertical.
    """
    if axis == 1:  # Horizontal
        return np.abs(image[:, :-1] - image[:, 1:])
    elif axis == 0:  # Vertical
        return np.abs(image[:-1, :] - image[1:, :])

def plot_pixel_distribution(image, title, row_index):
    """
    Compute and plot the distribution of pixel differences for horizontal and vertical adjacency.
    """
    diff_horizontal = compute_adjacent_pixel_difference(image, axis=1)
    diff_vertical = compute_adjacent_pixel_difference(image, axis=0)
    
    plt.subplot(2, 3, row_index * 3 + 2)
    plt.hist(diff_horizontal.flatten(), bins=256, color='purple')
    plt.title(f"{title} (Horizontal)")
    plt.xlabel("Pixel Difference")
    plt.ylabel("Frequency")
    if title=="Original":
        plt.ylim(0, 3500)  # Set y-axis limit to 3500

    plt.subplot(2, 3, row_index * 3 + 3)
    plt.hist(diff_vertical.flatten(), bins=256, color='purple')
    plt.title(f"{title} (Vertical)")
    plt.xlabel("Pixel Difference")
    plt.ylabel("Frequency")
    if title=="Original":
        plt.ylim(0, 3500)  # Set y-axis limit to 3500

@app.route('/plot', methods=['POST'])
def generate_plot():
    """
    Generate the plot for the original and encrypted images.
    Accepts two images via POST request as 'original_image' and 'encrypted_image'.
    """
    if 'original' not in request.files or 'cipher' not in request.files:
        return jsonify({"error": "Missing original or decrypted image"}), 400
    
    original_file = request.files['original']
    encrypted_file = request.files['cipher']
        
    # Read images
    original_image = cv2.imdecode(np.frombuffer(original_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    encrypted_image = cv2.imdecode(np.frombuffer(encrypted_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    plt.switch_backend('Agg')

    # Plot results
    plt.figure(figsize=(15, 10))    

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    plot_pixel_distribution(original_image, "Original", 0)

    # Encrypted image
    plt.subplot(2, 3, 4)
    plt.imshow(encrypted_image, cmap='gray')
    plt.title("Encrypted Image")
    plt.axis("off")
    plot_pixel_distribution(encrypted_image, "Encrypted", 1)

    plt.tight_layout()

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Return the plot as an image response
    return send_file(buffer, mimetype='image/png')


def get_adjacent_pixel_pairs(image, axis=1):
    """
    Extract adjacent pixel pairs for scatter plotting.
    :param image: 2D numpy array (grayscale image)
    :param axis: Direction of adjacent pixels (1 for horizontal, 0 for vertical)
    :return: Two lists of pixel intensities
    """
    if axis == 1:  # Horizontal adjacent pixels
        x = image[:, :-1].flatten()
        y = image[:, 1:].flatten()
    elif axis == 0:  # Vertical adjacent pixels
        x = image[:-1, :].flatten()
        y = image[1:, :].flatten()
    else:
        raise ValueError("Invalid axis. Use 0 for vertical or 1 for horizontal.")
    return x, y

@app.route('/plot1', methods=['POST'])
def plot1():
    """
    Generate scatter plots for adjacent pixel pairs in both original and encrypted images.
    """
    try:
        breakpoint()
        if 'original' not in request.files or 'cipher' not in request.files:
            return jsonify({"error": "Missing original or decrypted image"}), 400
        
        original_file = request.files['original']
        encrypted_file = request.files['cipher']
        original_image = cv2.imdecode(np.frombuffer(original_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        encrypted_image = cv2.imdecode(np.frombuffer(encrypted_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        original_horizontal_x, original_horizontal_y = get_adjacent_pixel_pairs(original_image, axis=1)
        encrypted_horizontal_x, encrypted_horizontal_y = get_adjacent_pixel_pairs(encrypted_image, axis=1)
        plt.figure(figsize=(10, 5))
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

        # Save the plot to a buffer
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)

        # Return the plot as a response
        return send_file(plot_buffer, mimetype='image/png', as_attachment=True, download_name='scatter_plot.png')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare-images', methods=['POST'])
def compare_images_endpoint():
    """Compare two uploaded images"""
    try:
        if 'original' not in request.files or 'cipher' not in request.files:
            return jsonify({"error": "Missing original or decrypted image"}), 400
        original_file = request.files['original']
        decrypted_file = request.files['cipher']
        original_img = Image.open(original_file)
        decrypted_img = Image.open(decrypted_file)
        original_rgb = original_img
        decrypted_rgb = decrypted_img
        # Perform comparison
        comparison_result = compare_images(original_rgb, decrypted_rgb)
        
        return jsonify(comparison_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/resource-usage', methods=['GET'])
def resource_usage_endpoint():
    """Get current system resource usage"""
    try:
        return jsonify(log_resource_usage())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "PRNG Oscillator Flask Server is running"
    })
def calculate_correlation(img1, img2):
    """Compute Pearson correlation coefficient."""
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    return np.corrcoef(img1_flat, img2_flat)[0, 1]

def calculate_entropy(image):
    """Compute Shannon entropy."""
    image = img_as_ubyte(image)  # Convert to 8-bit if necessary
    histogram, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    return entropy(histogram, base=2)
from skimage.color import rgb2gray
from skimage.io import imread

def preprocess_image(image, target_shape=None):
    """Ensure image is grayscale and resize to match target shape."""
    if image.ndim == 3:
        if image.shape[2] == 4:  # Handle RGBA images
            image = image[:, :, :3]  # Drop alpha channel
        image = rgb2gray(image)  # Convert to grayscale
    
    if target_shape:  # Resize if target shape is provided
        image = resize(image, target_shape, anti_aliasing=True)
    
    return image

@app.route('/corr-entropy', methods=['POST'])
def image_stats():
    # Load images from request

    original = imread(request.files['original'])
    cipher = imread(request.files['cipher'])
    decrypted = imread(request.files['decrypted'])
    target_shape = original.shape[:2]  
    original = preprocess_image(original,target_shape)
    cipher = preprocess_image(cipher,target_shape)
    decrypted = preprocess_image(decrypted,target_shape)

    # Convert to grayscale if needed
    original_gray = rgb2gray(original) if original.ndim == 3 else original
    cipher_gray = rgb2gray(cipher) if cipher.ndim == 3 else cipher
    decrypted_gray = rgb2gray(decrypted) if decrypted.ndim == 3 else decrypted

    # Compute values
    correlation_plain_cipher = calculate_correlation(original_gray, cipher_gray)
    correlation_plain_decrypted = calculate_correlation(original_gray, decrypted_gray)
    entropy_plain = calculate_entropy(original_gray)
    entropy_cipher = calculate_entropy(cipher_gray)

    # Prepare response
    response = {
        "size": f"{original.shape[0]} × {original.shape[1]}",
        "cc_plainvcipher": round(correlation_plain_cipher, 5),
        "e_plain": round(entropy_plain, 4),
        "e_cipher": round(entropy_cipher, 4),
        "cc_plainvsdecrypt": round(correlation_plain_decrypted, 5)
    }

    return jsonify(response)

@app.route('/corr-images', methods=['POST'])
def corr_images():
    # Load images from request

    original = imread(request.files['original'])
    cipher = imread(request.files['cipher'])
    target_shape = original.shape[:2]  
    original = preprocess_image(original,target_shape)
    cipher = preprocess_image(cipher,target_shape)

    # Convert to grayscale if needed
    original_gray = rgb2gray(original) if original.ndim == 3 else original
    cipher_gray = rgb2gray(cipher) if cipher.ndim == 3 else cipher

    # Compute values
    correlation_plain_cipher = calculate_correlation(original_gray, cipher_gray)

    # Prepare response
    response = {
        "cc_plainvcipher": round(correlation_plain_cipher, 5),
    }
    return jsonify(response)

def calculate_mse(original, encrypted):
    return np.mean((original.astype("float") - encrypted.astype("float")) ** 2)

def calculate_psnr(original, encrypted):
    mse = calculate_mse(original, encrypted)
    if mse == 0:
        return float('inf')  # No difference between images
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_npcr(original, encrypted):
    diff = np.abs(original - encrypted) > 0
    return np.sum(diff) / float(original.size) * 100

def calculate_uaci(original, encrypted):
    diff = np.abs(original.astype("float") - encrypted.astype("float"))
    return np.mean(diff) / 255 * 100
def resize_images(original, encrypted):
    # Resize encrypted image to match the original's dimensions
    return cv2.resize(encrypted, (original.shape[1], original.shape[0]))

@app.route('/formulas', methods=['POST'])
def evaluate_images():
    if 'original' not in request.files or 'cipher' not in request.files:
        return jsonify({"error": "Missing original or cipher image"}), 400

    original_file = request.files['original']
    encrypted_file = request.files['cipher']

    # Read the original and encrypted images
    original_image = cv2.imdecode(np.frombuffer(original_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    encrypted_image = cv2.imdecode(np.frombuffer(encrypted_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

    # Resize encrypted image to match the original's dimensions
    encrypted_image = resize_images(original_image, encrypted_image)

    # Calculate metrics
    mse_value = calculate_mse(original_image, encrypted_image)
    psnr_value = calculate_psnr(original_image, encrypted_image)
    npcr_value = calculate_npcr(original_image, encrypted_image)
    uaci_value = calculate_uaci(original_image, encrypted_image)

    # Return the computed metrics as JSON
    return jsonify({
        "MSE": mse_value,
        "PSNR": psnr_value,
        "NPCR": npcr_value,
        "UACI": uaci_value
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
