import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "age": [20, 25, np.nan, 30, np.nan],
    "salary": [30000, np.nan, 45000, 50000, 60000],
    "city": ["A", "B", np.nan, "A", "C"]
})
imputer=SimpleImputer(strategy='mean')
df[['age']]=imputer.fit_transform(df[['age']])
df[['salary']]=imputer.fit_transform(df[['salary']])


# imputer2=SimpleImputer(strategy='most_frequent')
# df['city']=imputer2.fit_transform(df[['city']])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[['city']] = cat_imputer.fit_transform(df[['city']])
print(df)


