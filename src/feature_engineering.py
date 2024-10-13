
import pandas as pd

# Load the dataset
df = pd.read_csv('data/Bank Customer Churn Prediction.csv')

import pandas as pd

# Feature engineering: creating age groups
def create_age_groups(df):
    bins = [18, 30, 45, 60, 100]  # Age ranges
    labels = ['Young', 'Adult', 'Middle-Aged', 'Senior']  # Labels for age groups
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
    return df

df = create_age_groups(df)

# Encoding categorical variables
def encode_categorical(df):
    # One-hot encoding the 'country' column
    df = pd.get_dummies(df, columns=['country','age_group'], drop_first=True)
    
    # Label encoding for 'gender'
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    return df

df = encode_categorical(df)

# Save the feature-engineered data to a new CSV file
df.to_csv('data/processed_bank_churn.csv', index=False)

print("Feature engineering completed and data saved to 'data/processed_bank_churn.csv'")
