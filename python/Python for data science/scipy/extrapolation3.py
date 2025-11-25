# import numpy as np
# from scipy.interpolate import interp1d 

# x = np.array([0, 10, 20])
# y = np.array([5, 25, 60])

# f=interp1d(x,y,kind='linear',fill_value='extrapolate')
# print(f(30))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, interp1d

# Example data points
x = np.array([0, 10, 20, 30])
y = np.array([5, 25, 15, 40])

# Create interpolators
linear_f = interp1d(x, y, kind="linear", fill_value="extrapolate")
spline_f = CubicSpline(x, y, extrapolate=True)

# Create fine x-values for smooth curve
x_fine = np.linspace(-10, 50, 400)

# Get interpolated y-values
y_linear = linear_f(x_fine)
y_spline = spline_f(x_fine)

# Plot both curves + original points
plt.figure(figsize=(10, 5))
plt.plot(x_fine, y_linear, label="Linear Interpolation")
plt.plot(x_fine, y_spline, label="Cubic Spline Interpolation")
plt.scatter(x, y, color="black", label="Data Points")
plt.title("Linear vs Spline Interpolation")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Test values
test_x = [5, 15, 25, 30]
print("Input x values:", test_x)
print("Linear interpolation:", linear_f(test_x))
print("Spline interpolation:", spline_f(test_x))

# Extrapolation example
print("\nExtrapolated at x = 50:")
print("Linear:", linear_f(50))
print("Spline:", spline_f(50))
