"""
Unit tests for model evaluation
Verifies that model performance metrics are calculated correctly
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TestModelTraining:
    """Tests for model training"""
    
    def test_model_trains_successfully(self):
        """Verify model trains without errors"""
        # Load data
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        assert model is not None
        print("✅ Model trained successfully")
    
    def test_model_makes_predictions(self):
        """Verify trained model can make predictions"""
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        assert len(predictions) == len(X_test)
        print(f"✅ Model made {len(predictions)} predictions")


class TestModelPredictions:
    """Tests for prediction validity"""
    
    def test_predictions_are_binary(self):
        """Verify predictions are 0 or 1"""
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        unique_predictions = np.unique(predictions)
        
        assert set(unique_predictions).issubset({0, 1})
        print(f"✅ Predictions are valid: {unique_predictions}")
    
    def test_probabilities_in_valid_range(self):
        """Verify predicted probabilities are [0, 1]"""
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        assert (probs >= 0).all() and (probs <= 1).all()
        print(f"✅ Probabilities in range [0, 1]")


class TestModelEvaluation:
    """Tests for evaluation metrics"""
    
    def test_evaluation_metrics_calculated(self):
        """Verify evaluation metrics are computed"""
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        auc_roc = roc_auc_score(y_test, probs)
        
        # Verify all metrics calculated
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= auc_roc <= 1
        
        print(f"✅ Metrics calculated:")
        print(f"   Accuracy: {accuracy:.2f}")
        print(f"   Precision: {precision:.2f}")
        print(f"   Recall: {recall:.2f}")
        print(f"   AUC-ROC: {auc_roc:.2f}")


class TestMetricValidity:
    """Tests for metric validity"""
    
    def test_metrics_in_valid_range(self):
        """Verify all metrics are between 0 and 1"""
        df = pd.read_csv('data/creditcard.csv')
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        auc_roc = roc_auc_score(y_test, probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc_roc': auc_roc
        }
        
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} out of range: {value}"
        
        print(f"✅ All metrics in valid range [0, 1]")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
