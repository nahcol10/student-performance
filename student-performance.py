#!/usr/bin/env python
# coding: utf-8

# # STAIML ASSIGNMENT 2

# NAME:LOCHAN PAUDEL
# PRN: 23070126170
# BRANCH:AIML(A3)
# BATCH:2023-27

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Read the data files
df_mat = pd.read_csv(r"C:\Users\DELL\OneDrive\Desktop\student\student-mat.csv")


# a)Identify which features have null values? How will you address the null values in different features?

# In[12]:


print("Features with null values:")
print(df_mat.isnull().sum())


# b) What transformations might be necessary for categorical variables such as 'school', 'sex', 'address', 
# 'famsize', and 'reason' before applying linear regression?

# In[ ]:


data = pd.get_dummies(df_mat, columns=['school', 'sex', 'address', 'famsize', 'reason'])


# c) Considering the numeric attributes like 'age', 'absences', and 'G1', 'G2' grades, would you perform any 
# normalization or scaling before fitting a linear regression model? If yes, what method would you choose 
# and why?

# In[15]:


scaler = StandardScaler()
df_mat[['age', 'absences', 'G1', 'G2']] = scaler.fit_transform(df_mat[['age', 'absences', 'G1', 'G2']])


# d) Which features have outliers present? How will you address those?

# In[18]:


sns.violinplot(data=data)
plt.xlabel('age')
plt.ylabel('absences')
plt.title('Violin Plot of Age, Absences, G1, G2')
plt.show()


# e) Would you create any new features (feature engineering) from existing ones, such as combining parental 
# education levels into a single feature or creating a feature representing total alcohol consumption?

# In[17]:


df_mat['parent_education'] = df_mat['Medu'] + df_mat['Fedu']
df_mat['total_alcohol'] = df_mat['Dalc'] + df_mat['Walc']


# f) Which features will you select for prediction of final grades (G3) and how?

# In[18]:


selected_features = ['age', 'absences', 'G1', 'G2', 'parent_education', 'total_alcohol', 'studytime']
X = df_mat[selected_features]
y = df_mat['G3']


# g) Since 'G1' and 'G2' grades are highly correlated with 'G3', would you consider dropping any of them to 
# avoid multicollinearity in the linear regression model? If yes, which one and why?

# In[19]:


corr_matrix = df_mat[['G1', 'G2', 'G3']].corr()
print(corr_matrix)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# h) How would you evaluate the performance of the linear regression model in predicting 'G3' grades? Which 
# metrics would you use, and why are they appropriate for this task

# In[22]:


print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R sq. Score:", r2_score(y_test, y_pred))

