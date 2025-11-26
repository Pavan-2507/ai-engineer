import matplotlib.pyplot as plt
import numpy as np
x=np.array([1,2,3,4,5,6])
y=x**2
plt.plot(x,y, marker='s')
plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# plt.scatter(x,y)
# plt.tight_layout()
plt.show()