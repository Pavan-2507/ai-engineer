#Installation command
#pip install scipy
import numpy as np
from scipy import signal, datasets
from scipy.interpolate import interp1d , CubicSpline
import matplotlib.pyplot as plt
# Depths in meters (known data points)
depths = np.array([1000, 1500, 2000, 2500, 3000]) 
# Pressure in psi (known data points corresponding to depths)
pressures = np.array([3500, 4000, 4500, 5000, 5500])
# Create an interpolation function using cubic spline
pressure_interpolation = interp1d(depths, pressures, kind='cubic')
# Depths at which we want to estimate pressure (unknown data points)
depths_to_interpolate = np.linspace(1000, 3000, 100) 
interpolated_pressures = pressure_interpolation(depths_to_interpolate)


# Plotting the original data points and the interpolated curve
plt.figure(figsize=(8,6))
plt.plot(depths_to_interpolate,interpolated_pressures , 'o', label='Original Data Points', color='red')
plt.title('Pressure vs Depth in an Oil Well- Without interpolation')
plt.xlabel('Depth (m)')
plt.ylabel('Pressure (psi)')
plt.legend()
plt.grid(True)
plt.show()
