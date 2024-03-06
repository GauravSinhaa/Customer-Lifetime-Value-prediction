#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = "https://docs.google.com/spreadsheets/d/1cltj1nQA2hSM_-BJ2b7S1afMV78hKj9ygd4_P-Aqjqk/export?format=csv"
data = pd.read_csv(url)

# Explore the dataset
print(data.head())
print(data.info())

# Preprocessing the data (handle missing values, encoding categorical variables, etc.)
# Splitting the data into features and target variable
X = data.drop(columns=['Customer Lifetime Value'])
y = data['Customer Lifetime Value']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training different regression models
# Example: Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("RMSE:", rmse)
print("R-squared:", r2)

# Visualization (optional)
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CLV")
plt.ylabel("Predicted CLV")
plt.title("Actual vs Predicted CLV")
plt.show()


# In[ ]:




