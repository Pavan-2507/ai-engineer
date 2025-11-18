import numpy as np
import time

# Python loop
nums = list(range(1_000_000))
start = time.time()
result = [n * 2 for n in nums]
print(start)
print("Python:", time.time() - start)

# NumPy
arr = np.arange(1_000_000)
start = time.time()
result = arr * 2
print(start)
print("NumPy:", time.time() - start)



# 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# 3D array
arr3d = np.array([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])

print(arr2d.ndim)   # 2
print(arr3d.ndim)   # 3

result1=np.zeros((3, 4))
print(result1)


print(np.full((2, 2), 7))


print(np.diag([10, 20, 30]))

print(np.random.randn(3, 3))

print(np.random.rand(2, 3))

print(np.random.seed(42))

print(np.full((2,2),5))

print(np.arange(0,10,2))

print(np.random.randint(1, 10, (2, 4)))
