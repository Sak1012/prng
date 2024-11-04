import io
import os

import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PIL import Image
from scipy.integrate import solve_ivp


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

        # Use entropy for the initial state
        entropy = os.urandom(32)  # Get 32 bytes of secure random bytes
        self.initial_state = [
            (b % 100) / 100.0 for b in entropy[:4]
        ]  # Normalize to [0, 1]

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
        bits = []
        for i in range(num_bits):
            chaotic_value = trajectory[0][i % len(trajectory[0])]
            bit = 1 if chaotic_value > 0 else 0
            bits.append(bit)

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

        data_to_encrypt = (
            len(format_bytes).to_bytes(1, "big") + format_bytes + image_bytes
        )

    iv = os.urandom(16)

    cipher = AES.new(key, AES.MODE_CBC, iv)

    padded_data = pad(data_to_encrypt, AES.block_size)
    encrypted_data = cipher.encrypt(padded_data)

    return iv + encrypted_data


def decrypt_image(encrypted_data, key, output_path=None):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]

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

def main():
    mhho_prng = PrngOscillator()

    t_span = (0, 2000)
    _, trajectory = mhho_prng.simulate(t_span)

    random_bits = mhho_prng.generate_prng_bits(trajectory)
    lfsr_bits = mhho_prng.lfsr_post_processing(random_bits)

    entropy = os.urandom(16)
    entropy_bits = np.frombuffer(entropy, dtype=np.uint8).reshape(-1) % 2

    extended_entropy_bits = np.tile(entropy_bits, 8)[:128]

    mixed_bits = (lfsr_bits + extended_entropy_bits) % 2
    key = bytes(
        int("".join(map(str, mixed_bits[i : i + 8])), 2) for i in range(0, 128, 8)
    )

    print("Key bits:", mixed_bits)

    encrypted_data = encrypt_image("catpix.jpg", key)

    with open("encrypted_image.bin", "wb") as f:
        f.write(encrypted_data)

    decrypted_img = decrypt_image(encrypted_data, key)
    decrypted_img.save("decrypted_image.jpg")

if __name__ == "__main__":
    main()
