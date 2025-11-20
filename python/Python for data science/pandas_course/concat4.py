import pandas as pd
data=pd.read_csv('marks_dataset.csv')
marks1=data.iloc[0:5]
marks2=data.iloc[6:10]
print(marks1)
print("pavan")
print(marks2)
data2=pd.concat([marks1,marks2],sort=False)
print(data2)
