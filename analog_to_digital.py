import numpy as np

class AnalogToDigitalConverter:
    def __init__(self, bits=8, v_min=-75, v_max=40):
        self.bits = bits
        self.v_min = v_min
        self.v_max = v_max
        self.levels = 2**bits
        self.step = (v_max - v_min) / self.levels

    def convert(self, analog_signal):
        clipped_signal = np.clip(analog_signal, self.v_min, self.v_max)
        digital_signal = np.round((clipped_signal - self.v_min) / self.step).astype(int)
        return digital_signal

    def convert_to_voltage(self, digital_signal):
        return digital_signal * self.step + self.v_min
