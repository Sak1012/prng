import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mhso_system import MHSOSystem
from data_handler import save_to_csv
from analog_to_digital import AnalogToDigitalConverter

def run_simulation():
    system = MHSOSystem()
    initial_state = [0.1, 0.1, 0.1, 0.1]
    t_span = (0, 2000)
    t_eval = np.linspace(0, 2000, 200000)

    solution = solve_ivp(system.equations, t_span, initial_state, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-9)

    return solution

def plot_results(analog_signal, digital_signal, adc):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(analog_signal[10000:], 'b-', label='Analog')
    plt.ylabel('Voltage', fontsize=14)
    plt.title('Analog Signal', fontsize=16)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.step(range(len(digital_signal[10000:])), adc.convert_to_voltage(digital_signal[10000:]), 'r-', label='Digital')
    plt.ylabel('Voltage', fontsize=14)
    plt.xlabel('Sample', fontsize=14)
    plt.title('Digital Signal', fontsize=16)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    solution = run_simulation()
    
    # Extract the voltage (x) from the solution
    analog_signal = solution.y[0]

    # Create an ADC instance and convert the signal
    adc = AnalogToDigitalConverter(bits=8, v_min=np.min(analog_signal), v_max=np.max(analog_signal))
    digital_signal = adc.convert(analog_signal)

    # Plot results
    plot_results(analog_signal, digital_signal, adc)

    # Save both analog and digital signals
    save_to_csv(np.column_stack((solution.y.T, digital_signal)), "MHSO_output_with_digital.csv")
