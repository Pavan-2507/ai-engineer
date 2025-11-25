import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# STEP 1: Create a time axis (t) and a signal
duration = 2.0           # seconds
fs = 500                 # sampling frequency (samples per second)
dt = 1.0 / fs            # time step between samples

t = np.arange(0, duration, dt)   # time array: 0, dt, 2*dt, ..., duration

# Make a signal with two sine waves + noise
freq1 = 5    # Hz
freq2 = 20   # Hz

signal = 1.5 * np.sin(2 * np.pi * freq1 * t) + \
         0.8 * np.sin(2 * np.pi * freq2 * t) + \
         0.5 * np.random.randn(len(t))  # add some noise

# (Optional) Plot the time-domain signal
plt.figure(figsize=(10, 3))
plt.plot(t, signal)
plt.title("Time Domain Signal")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 2: Compute the FFT of the signal
N = len(t)           # number of samples
yf = fft(signal)     # complex frequency spectrum

# STEP 3: Create the frequency axis
xf = fftfreq(N, dt)  # frequency values for each FFT bin

# STEP 4: Keep only the positive half of the spectrum
half_N = N // 2
xf_pos = xf[:half_N]
yf_pos = yf[:half_N]

# STEP 5: Compute magnitude (strength) of each frequency
spectrum = np.abs(yf_pos) / N   # normalize by N so scale is reasonable

# STEP 6: Plot the magnitude spectrum
plt.figure(figsize=(10, 3))
plt.plot(xf_pos, spectrum)
plt.title("Frequency Domain (Magnitude Spectrum)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# STEP 7: Print some key info
print("Sampling frequency fs:", fs, "Hz")
print("Number of samples N:", N)
print("Frequency resolution df:", xf[1] - xf[0], "Hz")

# Find top 5 peaks (largest frequency components)
indices_sorted = np.argsort(spectrum)[::-1]  # sort indices by descending amplitude
top_k = 5
print("\nTop frequency components:")
for i in range(top_k):
    idx = indices_sorted[i]
    print(f"Freq = {xf_pos[idx]:6.2f} Hz,  Amplitude ≈ {spectrum[idx]:.3f}")
