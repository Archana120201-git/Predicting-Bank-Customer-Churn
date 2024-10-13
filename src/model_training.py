# model_training.py

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the processed data
df = pd.read_csv('data/processed_bank_churn.csv')

# Define input features (X) and target (y)
X = df.drop(['churn','customer_id'], axis=1)  # Drop the target column 'churn' from features
y = df['churn']  # Define the target column 'churn'

# Train-test split (80% training data, 20% testing data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Prevents warning for label encoder
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation results
print(f"XGBoost - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")
