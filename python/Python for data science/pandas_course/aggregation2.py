import pandas as pd
import numpy as np
data=pd.read_csv('marks_dataset.csv')
# print(data.head())
data2=data.iloc[:,1:6]
data2['total']=data2.apply(np.sum,axis=1)
# print(data2.head())
print(data2['total'].sum())
print(data2['total'].mean())
data3=pd.read_csv('car_dataset.csv')
sentences = []
for i in  range (len(data3)):
    row=data3.iloc[i]
    sentence = (
        f"The {row['car_name']} is a {row['type']} manufactured by "
        f"{row['manufacturer']} in {row['model_year']}. "
        f"It costs ₹{row['cost']}, has a {row['engine_cc']} cc engine "
        f"and gives {row['mileage']} kmpl mileage."
    )

    sentences.append(sentence)

data3['sentence']=sentences
print(data3.iloc[1:5,1:10].head())

for i in range (len(sentences)):
    print(sentences[i])
