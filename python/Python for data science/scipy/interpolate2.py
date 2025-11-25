import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import  interp1d
x=np.array([0,1,2,3,4])
y=np.array([0,2,4,6,8])
f=interp1d(x,y,kind='linear')
print(f(1.5))

f1 = interp1d(x, y, kind='cubic')
print(f1(1.5))


depths = np.array([0, 10, 20, 30])
pressure = np.array([1, 5, 18, 40])
f2=interp1d(depths,pressure,kind='cubic')
print(f2(15))
plt.plot(x,y,label="Linear Interpolation")
plt.show()


