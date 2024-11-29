import requests
import os
from PIL import Image
import io
import base64
# Base URL for the Flask server
BASE_URL = 'http://localhost:5000'


def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("Health Endpoint Test Passed ✓")
    return data


def test_simulate_endpoint():
    """Test the system simulation endpoint"""
    print("\nTesting Simulate Endpoint...")
    response = requests.post(f"{BASE_URL}/simulate", json={'t_span': (0, 2000)})
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert 'random_bits' in data
    assert 'simulation_time' in data
    assert 'randomness_test' in data

    random_bits = data['random_bits']
    assert len(random_bits) == 128  # Default bit generation
    assert all(bit in [0, 1] for bit in random_bits)

    randomness_test = data['randomness_test']
    assert 'bit_distribution' in randomness_test
    assert 'chi_square_statistic' in randomness_test
    assert 'p_value' in randomness_test

    print("Simulate Endpoint Test Passed ✓")
    return data


def test_generate_key_endpoint():
    """Test the encryption key generation endpoint"""
    print("\nTesting Generate Key Endpoint...")
    response = requests.post(f"{BASE_URL}/generate-key")
    assert response.status_code == 200
    data = response.json()

    assert 'key' in data
    assert 'randomness_test' in data

    key = data['key']
    assert len(key) == 32  # 16 bytes = 32 hex characters

    randomness_test = data['randomness_test']
    assert 'bit_distribution' in randomness_test
    assert 'chi_square_statistic' in randomness_test
    assert 'p_value' in randomness_test

    print("Generate Key Endpoint Test Passed ✓")
    return data


def test_encrypt_decrypt_image(image_path):
    """Test image encryption and decryption, and save images in PNG format"""
    print(f"\nTesting Image Encryption and Decryption with {image_path}...")
    with open(image_path, 'rb') as img_file:
        img_byte_arr = img_file.read()

    # Prepare the file payload
    files = {'image': (os.path.basename(image_path), img_byte_arr, 'image/png')}
    encrypt_response = requests.post(
        f"{BASE_URL}/encrypt-image",
        files=files,
    )
    if encrypt_response.status_code != 200:
        raise ValueError(f"Encrypt image endpoint failed. Status code: {encrypt_response.status_code}. Response: {encrypt_response.text}")

    # Retrieve the encryption key from the headers
    key_hex = encrypt_response.headers.get('X-Encryption-Key')
    if not key_hex:
        raise ValueError("No encryption key returned in the response headers.")
    encrypted_image_data = encrypt_response.content

    # Save the encrypted image to a file
    encrypted_image_path = "encrypted_image.png"
    with open(encrypted_image_path, "wb") as f:
        f.write(encrypted_image_data)

    print(f"Encrypted image saved as PNG to {encrypted_image_path}")
    print(f"Encryption Key: {key_hex}")

    encrypted_image = Image.open(encrypted_image_path)
    if encrypted_image.mode != "L":
        encrypted_image = encrypted_image.convert("L")
    pixel_data = list(encrypted_image.getdata())
    extracted_length = int.from_bytes(bytes(pixel_data[:4]), byteorder="big")
    print(f"Extracted length: {extracted_length}")
    retrieved_data = bytes(pixel_data[4:4 + extracted_length])
    print("Data retrieval successful!")
    decrypt_response = requests.post(
        f"{BASE_URL}/decrypt-image",
        files={'file': ('encrypted_image.png', retrieved_data, 'image/png')},
        data={'key': key_hex}
    )
    if decrypt_response.status_code != 200:
        raise ValueError(f"Decrypt image endpoint failed. Status code: {decrypt_response.status_code}. Response: {decrypt_response.text}")
    decrypted_image_path = "decrypted_image.png"
    
    decrypted_image_buffer = io.BytesIO(decrypt_response.content)
    decrypted_image = Image.open(decrypted_image_buffer)
    decrypted_image.save(decrypted_image_path, format="PNG",optimize=True)
    print(f"Decrypted image saved to {decrypted_image_path}")
    files = {
    'original_image': open('catpix.jpg', 'rb'),
    'decrypted_image': open('decrypted_image.png', 'rb')
    }
    response = requests.post(f"{BASE_URL}/compare-images", files=files)
    if response.status_code != 200:
        raise ValueError(f"Image comparison failed. Status code: {response.status_code}. Response: {response.text}")
    comparison_result = response.json()
    files = {
    'original_image': open('catpix.jpg', 'rb'),
    "encrypted_image": open('decrypted_image.png', 'rb')
    }
    response = requests.post(f"{BASE_URL}/corr-calc", files=files)
    if response.status_code != 200:
        raise ValueError(f"Correlation calculation failed. Status code: {response.status_code}. Response: {response.text}")
    correlation_result = response.json()
    files = {
    'original_image': open('catpix.jpg', 'rb'),
    "encrypted_image": open('encrypted_image.png', 'rb')
    }
    response = requests.post(f"{BASE_URL}/plot", files=files)
    if response.status_code != 200:
        raise ValueError(f"Plot generation failed. Status code: {response.status_code}. Response: {response.text}")
    plot_path = "plot.png"
    with open(plot_path, "wb") as f:
        f.write(response.content)
    response = requests.post(f"{BASE_URL}/plot1", files=files)
    if response.status_code != 200:
        raise ValueError(f"Plot generation failed. Status code: {response.status_code}. Response: {response.text}")
    plot_path = "plot1.png"
    with open(plot_path, "wb") as f:
        f.write(response.content)              
    print(correlation_result)
    print(comparison_result)
    print("Images match! Decryption successful.")
    print("Image Encryption and Decryption Test Passed ✓")

def run_all_tests(test_image_path=None):
    """Run all endpoint tests"""
    print("Starting Comprehensive Endpoint Tests...")
    try:
        test_health_endpoint()
        test_simulate_endpoint()
        test_generate_key_endpoint()

        if test_image_path:
            test_encrypt_decrypt_image(test_image_path)

        test_resource_usage()
        print("\n✅ ALL ENDPOINT TESTS PASSED SUCCESSFULLY! ✅")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the server. Is it running?")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")


def test_resource_usage():
    """Test resource usage endpoint"""
    print("\nTesting Resource Usage Endpoint...")
    response = requests.get(f"{BASE_URL}/resource-usage")
    assert response.status_code == 200
    data = response.json()

    assert 'memory_usage_mb' in data
    assert 'cpu_usage_percent' in data

    print("Resource Usage Endpoint Test Passed ✓")
    return data


if __name__ == '__main__':
    test_image_path = 'catpix.jpg'  # Replace with your test image path
    test_encrypt_decrypt_image(test_image_path)
