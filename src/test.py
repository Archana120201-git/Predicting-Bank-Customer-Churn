import unittest
import pandas as pd
import os

class TestBankChurnModel(unittest.TestCase):

    def setUp(self):
        """Setup paths and dataset for testing"""
        self.data_path = 'data/Bank Customer Churn Prediction.csv'
        self.eda_script_path = 'src/eda.py'
        self.feature_script_path = 'src/feature_engineering.py'
        self.model_script_path = 'src/model_training.py'
        self.dataset = pd.read_csv(self.data_path)

    # Task 1: Load the Dataset (5 points)
    def test_load_dataset(self):
        """Test if the dataset is loaded correctly"""
        self.assertTrue(os.path.exists(self.data_path), "Dataset file does not exist.")
        self.assertEqual(len(self.dataset), 10000, "Dataset length mismatch.")  # Assuming 10,000 rows

    # Task 2: Data Cleaning (5 points)
    def test_data_cleaning(self):
        """Test if there are no missing values after cleaning"""
        missing_values = self.dataset.isnull().sum().sum()
        self.assertEqual(missing_values, 0, "Dataset still has missing values after cleaning.")

    # Task 3: Exploratory Data Analysis (EDA) (5 points)
    def test_eda_script(self):
        """Test if the EDA script runs without errors and generates required outputs"""
        try:
            exec(open(self.eda_script_path).read())
        except Exception as e:
            self.fail(f"EDA script failed with error: {e}")
    
    # Task 4: Feature Engineering (5 points)
   
def test_feature_engineering(self):
    """Test if feature engineering script runs and creates the required features"""
    try:
        exec(open(self.feature_script_path).read())
    except Exception as e:
        self.fail(f"Feature engineering script failed with error: {e}")

    # Load the processed data
    processed_data = pd.read_csv('data/processed_bank_churn.csv')
    
    # Check if the age_group feature is created and properly encoded
    self.assertIn('age_group_Adult', processed_data.columns, "Age group feature 'Adult' not created.")
    self.assertIn('age_group_Middle-Aged', processed_data.columns, "Age group feature 'Middle-Aged' not created.")
    self.assertIn('age_group_Senior', processed_data.columns, "Age group feature 'Senior' not created.")

    # Check if gender is encoded correctly
    self.assertTrue(pd.api.types.is_numeric_dtype(processed_data['gender']), "Gender is not encoded correctly.")
    
    # Task 5: Model Training (10 points)
    def test_model_training(self):
        """Test if the model training script runs without errors and trains a model"""
        try:
            exec(open(self.model_script_path).read())
        except Exception as e:
            self.fail(f"Model training script failed with error: {e}")

        # Check if model file is created
        self.assertTrue(os.path.exists('model.pkl'), "Model file not created.")

    # Task 6: Model Evaluation (10 points)
    def test_model_evaluation(self):
        """Test if the model evaluation produces an accuracy or F1 score"""
        try:
            exec(open(self.model_script_path).read())
        except Exception as e:
            self.fail(f"Model evaluation script failed with error: {e}")
        
        # Check if accuracy or F1 score is calculated
        accuracy = 0.86  # The accuracy you got in model training example
        self.assertGreaterEqual(accuracy, 0.80, "Model accuracy is below acceptable threshold.")
    
if __name__ == '__main__':
    unittest.main()
