import pandas as pd
# import numpy as np
data=pd.read_csv('marks_dataset.csv')
print(data.head())
data2=data.iloc[:,0:10]

print(data2.agg({'total':['sum','mean','max']}))
print("total sum :"+ f"{data2['total'].sum()}")


print(" average : "+f"{data2.groupby('name')[''].mean()}")
