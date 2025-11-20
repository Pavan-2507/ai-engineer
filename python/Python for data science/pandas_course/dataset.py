import pandas as pd
data=pd.read_csv('car_dataset.csv')
# print(data.head())
# print(data.iloc[1:10,1:10])
list1=[col for col in data.columns if data[col].dtype in ['float','int']]
print(data[list1].agg(['min','max']))