import pandas as pd
data=pd.read_csv("ecommerce_dataset.csv")
print(data.iloc[:10])
print(data.shape)