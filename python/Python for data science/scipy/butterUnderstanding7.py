from scipy.signal import butter,filtfilt
import numpy as np
import matplotlib.pyplot as plt

t=np.linspace(0,1,500)
signal = np.sin(2*np.pi*5*t) + 0.5*np.random.randn(500)
b,a=butter(N=3,Wn=0.1)
smooth_signal=filtfilt(b,a,signal)
plt.plot(t, signal)          # noisy
plt.plot(t, smooth_signal)    # filtered
plt.show()

