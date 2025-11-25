from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 1, 500)   # 1 second, 500 samples
signal = 2*np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t)
yf = fft(signal)                 # frequency domain
xf = fftfreq(len(t), t[1]-t[0])  

print(yf)


print("************************************************************")


print(xf)
plt.plot(xf[:250], np.abs(yf[:250]))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()
