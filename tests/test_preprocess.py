"""
Unit tests for data preprocessing pipeline
Verifies that preprocessing works correctly
"""

import pytest
import pandas as pd
import numpy as np


class TestDataLoading:
    """Tests for data loading"""
    
    def test_data_loads_successfully(self):
        """Verify CSV loads without errors"""
        df = pd.read_csv('data/creditcard.csv')  # Adjust path as needed
        assert df is not None
        assert len(df) > 0
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    def test_data_has_required_columns(self):
        """Verify dataset has required columns"""
        df = pd.read_csv('data/creditcard.csv')
        required_cols = ['Time', 'V1', 'V2', 'Amount', 'Class']  # Adjust as needed
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        print(f"✅ All required columns present")


class TestMissingValues:
    """Tests for missing value handling"""
    
    def test_no_missing_values_in_data(self):
        """Verify dataset has no missing values"""
        df = pd.read_csv('data/creditcard.csv')
        missing = df.isnull().sum().sum()
        assert missing == 0, f"Found {missing} missing values"
        print(f"✅ No missing values in dataset")


class TestFeatureScaling:
    """Tests for feature scaling"""
    
    def test_features_are_scaled_correctly(self):
        """Verify numerical features are in reasonable range"""
        df = pd.read_csv('data/creditcard.csv')
        
        # Drop target column
        X = df.drop('Class', axis=1)
        
        # Check that features are not too large
        max_val = X.max().max()
        min_val = X.min().min()
        
        assert max_val < 100000, f"Features too large: max={max_val}"
        assert min_val > -100000, f"Features too small: min={min_val}"
        print(f"✅ Features in reasonable range [{min_val:.2f}, {max_val:.2f}]")


class TestTargetDistribution:
    """Tests for target variable"""
    
    def test_target_is_binary(self):
        """Verify target is binary classification"""
        df = pd.read_csv('data/creditcard.csv')
        unique_values = df['Class'].unique()
        
        assert len(unique_values) == 2, f"Expected 2 classes, got {len(unique_values)}"
        assert set(unique_values) == {0, 1}, "Classes should be 0 and 1"
        print(f"✅ Target is binary: {unique_values}")
    
    def test_target_imbalance_detected(self):
        """Verify dataset is imbalanced (expected for fraud)"""
        df = pd.read_csv('data/creditcard.csv')
        class_dist = df['Class'].value_counts()
        
        # Fraud datasets are typically 99%+ non-fraud
        non_fraud_ratio = class_dist[0] / len(df)
        fraud_ratio = class_dist[1] / len(df)
        
        print(f"✅ Class distribution - Non-fraud: {non_fraud_ratio:.1%}, Fraud: {fraud_ratio:.1%}")
        
        # Verify imbalance exists
        assert fraud_ratio < 0.5, "Dataset doesn't appear imbalanced (fraud detection)"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
