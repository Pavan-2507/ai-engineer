import numpy as np
from scipy.interpolate import interp1d , CubicSpline
import matplotlib.pyplot as plt
x = np.array([0, 10, 20, 30])
y = np.array([5, 25, 15, 40])


# Depths in meters (known data points)
depths = np.array([1000, 1500, 2000, 2500, 3000]) 
# Pressure in psi (known data points corresponding to depths)
pressures = np.array([3500, 4000, 4500, 5000, 5500])

spline_f=CubicSpline(depths,pressures,extrapolate=True)
x_fine=np.linspace(1000,3000,100)
y_linear=(x_fine)
plt.figure(figsize=(10,5))
plt.plot(x_fine,y_linear,label="Cubic Spline Interpolation")
plt.scatter(x, y, color="black", label="Data Points")

plt.title("Linear vs Spline Interpolation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

