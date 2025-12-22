# Linear Regression Example (House Price Prediction)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create sample dataset
size = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
price = np.array([40,   55,   65,   90,   100])  # in lakhs

# Step 2: Train Linear Regression Model
model = LinearRegression()
model.fit(size, price)

# Step 3: Print learned parameters
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Step 4: Predict for new value
new_size = np.array([[2001,2010,2050,2100]]).reshape(-1,1)
predicted_price = model.predict(new_size)
print("Predicted price for 1800 sqft:", predicted_price, "Lakhs")

# Step 5: Plot the data & line
plt.scatter(size, price, color='blue', label='Actual data')
plt.plot(size,model.predict(size), color='red', label='Regression Line')
plt.xlabel('Size (sqft)')
plt.ylabel('Price (Lakhs)')
plt.title('Linear Regression - House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()





