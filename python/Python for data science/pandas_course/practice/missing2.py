import pandas as pd
data=pd.read_csv("ecommerce_dataset.csv")
print(data.isnull().sum()) # find count of null values for each column
print(data.isnull().sum().idxmax()) # find highest null values column
print(data['product'].unique())
print(data.describe(include=object))
print(data.head())

data['quantity']=data['quantity'].fillna(data['quantity'].median())