import io
import os
import time
import psutil
import numpy as np
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from PIL import Image, ImageChops
from scipy.integrate import solve_ivp
from scipy.stats import chisquare
import random
class PrngOscillator:
    def __init__(self):
        self.params = {
            "g1": 1925.0,
            "gkv": 1700.0,
            "gL": 7.0,
            "v1": 100.0,
            "vk": -75.0,
            "vL": -40.0,
            "vc": 100.0,
            "kc": 3.3 / 18.0,
            "rao": 0.27,
            "lamn": 230.0,
            "gkc": 12.0,
            "k0": 0.1,
            "k1": 1.0,
            "k2": 0.5,
            "E": 0.0,
            "ohm1": 1.414,
            "ohm2": 0.618,
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


def miller_rabin(n, k=5):
    """Miller-Rabin primality test to check if a number is prime."""
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generate_rsa_keys(bits, prng_bits):
    p = int.from_bytes(prng_bits[:bits // 16], byteorder='big')  #
    q = int.from_bytes(prng_bits[bits // 16:bits // 8], byteorder='big')  # Use another portion for q

    # Make sure p and q are large primes
    while not miller_rabin(p):
        p += 1
    while not miller_rabin(q):
        q += 1  
    n = p * q
    e = 65537

    phi = (p - 1) * (q - 1)
    d = mod_inverse(e, phi)

    # Construct the RSA key using n, e, d, p, q
    rsa_key = RSA.construct((n, e, d, p, q))
    return rsa_key.publickey(), rsa_key


def mod_inverse(a, m):
    """Calculate the modular inverse of a under modulo m."""
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1


def is_prime(n):
    """Check if a number is prime (simple check for large primes)."""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def mod_inverse(a, m):
    """Calculate the modular inverse of a under modulo m."""
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def encrypt_image_aes(image_path, aes_key):
    with Image.open(image_path) as img:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=img.format or "PNG")
        image_bytes = img_buffer.getvalue()

    cipher = AES.new(aes_key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(image_bytes, AES.block_size))
    return cipher.iv + ciphertext  # prepend the IV to the ciphertext


def decrypt_image_aes(encrypted_data, aes_key, output_path=None):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)

    img = Image.open(io.BytesIO(decrypted_data))
    if output_path:
        img.save(output_path, format=img.format or "PNG")
    return img


def encrypt_aes_key_with_rsa(aes_key, public_key):
    cipher_rsa = PKCS1_OAEP.new(public_key)
    encrypted_aes_key = cipher_rsa.encrypt(aes_key)
    return encrypted_aes_key


def decrypt_aes_key_with_rsa(encrypted_aes_key, private_key):
    cipher_rsa = PKCS1_OAEP.new(private_key)
    aes_key = cipher_rsa.decrypt(encrypted_aes_key)
    return aes_key


def visually_compare_images(original_path, decrypted_path, blended_path="blend.jpg", diff_path="difference.jpg"):
    with Image.open(original_path) as original, Image.open(decrypted_path) as decrypted:
        print("\nVisual Comparison Results:")

        # Check for identical dimensions
        if original.size != decrypted.size:
            print(f"  - The images have different dimensions: {original.size} vs {decrypted.size}")
            return

        # Convert both images to RGB
        original_rgb = original.convert("RGB")
        decrypted_rgb = decrypted.convert("RGB")

        blend = Image.blend(original_rgb, decrypted_rgb, alpha=0.5)
        blend.save(blended_path)
        print(f"  - Blended image saved at {blended_path}. Visually inspect this to check for alignment issues.")

        diff = ImageChops.subtract(original_rgb, decrypted_rgb)
        diff.save(diff_path)
        print(f"  - Difference heatmap saved at {diff_path}. Areas with changes are highlighted.")


def compare_images(original_path, decrypted_path):
    with Image.open(original_path) as original, Image.open(decrypted_path) as decrypted:
        # Compare pixel-by-pixel difference
        diff = ImageChops.difference(original, decrypted)
        diff_bbox = diff.getbbox()

        print("\nImage Comparison Results:")
        if diff_bbox:
            print("  - The images are NOT identical.")
        else:
            print("  - The images are identical.")

        # Compare file sizes
        original_size = os.path.getsize(original_path)
        decrypted_size = os.path.getsize(decrypted_path)
        print(f"  - Original Image Size: {original_size / 1024:.2f} KB")
        print(f"  - Decrypted Image Size: {decrypted_size / 1024:.2f} KB")

        if original_size != decrypted_size:
            print("File sizes differ, due to added data during encryption.")
        else:
            print("  - File sizes match.")

        return diff


def log_resource_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024**2:.2f} MB")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")


def test_randomness(bits):
    counts = np.bincount(bits, minlength=2)
    expected = np.full_like(counts, len(bits) / 2)
    chi2, p_value = chisquare(counts, expected)
    print(
        f"\nRNG Randomness Test:\n"
        f"  - Distribution of bits: 0s={counts[0]}, 1s={counts[1]}\n"
        f"  - Chi-Square Statistic: {chi2:.2f}\n"
        f"  - p-value: {p_value:.6f} (p > 0.05 suggests randomness)"
    )
    return chi2, p_value


def main():
    print("Initializing the Memristive Helmholtz Oscillator...")
    mhho_prng = PrngOscillator()

    print("Simulating the chaotic system...")
    t_span = (0, 2000)
    start_time = time.time()
    _, trajectory = mhho_prng.simulate(t_span)
    rng_time = time.time() - start_time
    print(f"  - RNG Simulation completed in {rng_time:.6f} seconds.")

    print("\nGenerating PRNG bits...")
    prng_bits = mhho_prng.generate_prng_bits(trajectory, num_bits=128)

    print("\nApplying LFSR post-processing...")
    processed_bits = mhho_prng.lfsr_post_processing(prng_bits)

    print("\nGenerating RSA keys using PRNG bits...")
    public_key, private_key = generate_rsa_keys(2048, processed_bits)

    print("\nGenerating AES key using PRNG bits...")
    aes_key = processed_bits[:32]  # Use the first 256 bits from PRNG for the AES key

    print("\nEncrypting AES key with RSA...")
    encrypted_aes_key = encrypt_aes_key_with_rsa(aes_key, public_key)

    print("\nEncrypting image with AES...")
    original_image_path = "test_image.jpg"
    encrypted_image_data = encrypt_image_aes(original_image_path, aes_key)

    print("\nDecrypting AES key with RSA...")
    decrypted_aes_key = decrypt_aes_key_with_rsa(encrypted_aes_key, private_key)

    print("\nDecrypting image with AES...")
    decrypted_image = decrypt_image_aes(encrypted_image_data, decrypted_aes_key, output_path="decrypted_image.jpg")

    print("\nComparing images...")
    compare_images(original_image_path, "decrypted_image.jpg")
    visually_compare_images(original_image_path, "decrypted_image.jpg")

    print("\nLogging resource usage...")
    log_resource_usage()

    print("\nTesting randomness of PRNG output...")
    chi2, p_value = test_randomness(processed_bits)


if __name__ == "__main__":
    main()
