import pandas as pd
data=pd.read_csv("practice_missing.csv")
print(data.head())
# print(data.isnull().sum())
# print(data.count())
data1=data.dropna()
print(data1.count())
data1['total']=data1.iloc[:,3:6].sum(axis=1)
# print(data1.iloc[0:100])
data1.rename(columns={'maths':'Math', 'science':'Sci'},inplace=True)
# print(data1.head())
# data1.sort_values('total',ascending=False)
# print(data1.sort_values(['age','total'],ascending=[True,False]))
# print(data1.duplicated())
data1['gender'].replace({'Male':'M','Female':'F'},inplace=True)
print(data1.head())
marks=data1.iloc[:,3:6]
print(marks)
list1=[col for col in marks.columns if marks[col].dtype in ['float64','int64']]
print(list1)
for sub in list1:
    data1[sub]=data1[sub].apply(lambda x:x if x> 35 else "Fail")


data1['result']=data1[list1].apply(lambda row:"Fail" if "Fail" in row.values else "Pass",axis=1)

print(data1.head())

 