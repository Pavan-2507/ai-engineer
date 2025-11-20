import pandas as pd

# Employees and Groups
df_groups = pd.DataFrame({
    "Employee": ["Jyoti", "Sapna", "Raj", "Ramaswamy"],
    "Group": ["Accounting", "Engineering", "Engineering", "HR"]
})

# Employees and Hire Dates
df_hire = pd.DataFrame({
    "Employee": ["Jyoti", "Sapna", "Raj", "Ramaswamy"],
    "Hire_Date": [2004, 2008, 2012, 2014]
})

df3=pd.merge(df_groups,df_hire)
print(df3)

