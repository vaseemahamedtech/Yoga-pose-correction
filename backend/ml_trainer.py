"""
Machine Learning Trainer Module
Implements traditional ML algorithms: Random Forest, SVM, XGBoost, etc.
"""

import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available. Install with: pip install xgboost")

class MLTrainer:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0

    def initialize_models(self):
        """Initialize all ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'SVM': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=15,
                min_samples_split=5,
                random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                multi_class='multinomial',
                random_state=42,
                n_jobs=-1
            )
        }

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )

    def load_data(self, data_dir='dataset/processed'):
        """Load preprocessed data"""
        X_train = np.load(f'{data_dir}/X_train.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        X_val = np.load(f'{data_dir}/X_val.npy')
        y_val = np.load(f'{data_dir}/y_val.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')

        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)

        return (X_train, y_train_encoded), (X_val, y_val_encoded), (X_test, y_test_encoded)

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all models and evaluate on validation set"""
        results = {}

        print("\n🚀 Training ML Models...")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")

            # Train
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)

            print(f"   ✅ Validation Accuracy: {accuracy:.4f}")

            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred
            }

            # Track best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name

        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"   Accuracy: {self.best_accuracy:.4f}")

        return results

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Detailed evaluation of a model"""
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n📈 {model_name} Test Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_pred,
            target_names=self.label_encoder.classes_
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }

    def save_model(self, model, model_name, output_dir='models'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)

        model_path = f'{output_dir}/{model_name.replace(" ", "_")}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save label encoder
        encoder_path = f'{output_dir}/label_encoder.pkl'
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)

        print(f"✅ Model saved to {model_path}")

    def train_and_evaluate_all(self):
        """Complete ML training pipeline"""
        # Load data
        print("📥 Loading data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_data()

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Number of classes: {len(self.label_encoder.classes_)}")

        # Initialize and train models
        self.initialize_models()
        results = self.train_all_models(X_train, y_train, X_val, y_val)

        # Evaluate best model on test set
        print("\n" + "=" * 60)
        print("🎯 Final Evaluation on Test Set")
        print("=" * 60)

        test_results = self.evaluate_model(
            self.best_model,
            X_test,
            y_test,
            self.best_model_name
        )

        # Save best model
        self.save_model(self.best_model, f'best_ml_{self.best_model_name}')

        return {
            'validation_results': results,
            'test_results': test_results,
            'best_model': self.best_model_name,
            'best_accuracy': test_results['accuracy']
        }

if __name__ == "__main__":
    trainer = MLTrainer()

    # Check if processed data exists
    if not os.path.exists('dataset/processed/X_train.npy'):
        print("❌ Processed data not found!")
        print("   Please run data_preprocessor.py first")
    else:
        results = trainer.train_and_evaluate_all()
        print("\n✅ ML Training Complete!")
