import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt         #  importing some basic libraries to work with the dataset and for visualization

# Load the dataset
df = pd.read_csv('data/Bank Customer Churn Prediction.csv')

print(df.isnull().sum())                # check for missing values

print(df.describe())                    # generating summary statistics

print(df.mode())                        # check the mode value

# Visualize the distribution of Age
sns.histplot(df['age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Visualize the distribution of Balance
sns.histplot(df['balance'], kde=True)
plt.title('Distribution of Balance')
plt.show()

# Visualize the distribution of Credit Score
sns.histplot(df['credit_score'], kde=True)
plt.title('Distribution of Credit Score')
plt.show()

# Visualize the distribution of Estimated Salary
sns.histplot(df['estimated_salary'], kde=True)
plt.title('Distribution of Estimated Salary')
plt.show()
