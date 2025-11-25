import numpy as np
import pandas as pd
from scipy import signal,datasets
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Depths in meters (known data points)
depths = np.array([1000, 1500, 2000, 2500, 3000]) 

# Pressure in psi (known data points corresponding to depths)
pressures = np.array([3500, 4000, 4500, 5000, 5500])

# Plotting the original data points and the interpolated curve
plt.figure(figsize=(8,6))


pressure_interpolation= interp1d(depths, pressures, kind='linear')

depths_to_interpolate = np.linspace(1000, 3000, 100) 
interpolated_pressures = pressure_interpolation(depths_to_interpolate)
plt.plot(depths, pressures, 'o', label='Original')
plt.plot(depths_to_interpolate, interpolated_pressures, '-', label='Linear Interpolation')
plt.title('Pressure vs Depth in an Oil Well - With interpolation')
plt.xlabel('Depth (m)')
plt.ylabel('Pressure (psi)')
# plt.legend()
plt.grid(True)
plt.legend()
plt.show()


