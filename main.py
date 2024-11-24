import io
import os
import time
import psutil
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PIL import Image, ImageChops
from scipy.integrate import solve_ivp
from scipy.stats import chisquare


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


def encrypt_image(image_path, key):
    with Image.open(image_path) as img:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format=img.format or "PNG")
        image_bytes = img_buffer.getvalue()
        img_format = img.format or "PNG"
        format_bytes = img_format.encode("utf-8")
        data_to_encrypt = len(format_bytes).to_bytes(1, "big") + format_bytes + image_bytes
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data_to_encrypt, AES.block_size)
    return iv + cipher.encrypt(padded_data)


def decrypt_image(encrypted_data, key, output_path=None):
    iv, ciphertext = encrypted_data[:16], encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    decrypted_data = unpad(decrypted_padded, AES.block_size)
    format_length = decrypted_data[0]
    img_format = decrypted_data[1 : format_length + 1].decode("utf-8")
    image_data = decrypted_data[format_length + 1 :]
    img = Image.open(io.BytesIO(image_data))
    if output_path:
        img.save(output_path, format=img_format)
    return img

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

    print("\nGenerating PRNG bits and performing post-processing...")
    random_bits = mhho_prng.generate_prng_bits(trajectory)
    lfsr_bits = mhho_prng.lfsr_post_processing(random_bits)
    test_randomness(lfsr_bits)

    print("\nGenerating AES encryption key from PRNG bits...")
    entropy = os.urandom(16)
    entropy_bits = np.frombuffer(entropy, dtype=np.uint8).reshape(-1) % 2
    extended_entropy_bits = np.tile(entropy_bits, 8)[:128]
    mixed_bits = (lfsr_bits + extended_entropy_bits) % 2
    key = bytes(
        int("".join(map(str, mixed_bits[i : i + 8])), 2) for i in range(0, 128, 8)
    )
    print(f"  - Generated Key: {key.hex()}")

    print("\nEncrypting image...")
    encrypted_data = encrypt_image("catpix.jpg", key)

    print("\nDecrypting image...")
    decrypted_img = decrypt_image(encrypted_data, key, output_path="decrypted_image.jpg")

    print("\nComparing images for quality and discrepancies...")
    compare_images("catpix.jpg", "decrypted_image.jpg")
    visually_compare_images("catpix.jpg", "decrypted_image.jpg")
    print("\nLogging resource usage...")
    log_resource_usage()


if __name__ == "__main__":
    main()
