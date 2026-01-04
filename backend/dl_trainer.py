"""
Deep Learning Trainer Module
Implements CNN, LSTM, and Transfer Learning models
"""

import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class DLTrainer:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None

    def load_data(self, data_dir='dataset/processed'):
        """Load and prepare data for DL"""
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

        # Convert to categorical
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = to_categorical(y_train_encoded, num_classes)
        y_val_cat = to_categorical(y_val_encoded, num_classes)
        y_test_cat = to_categorical(y_test_encoded, num_classes)

        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat), num_classes

    def create_dense_nn_model(self, input_shape, num_classes):
        """Create Deep Neural Network with Dense layers"""
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),

            # First block
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Second block
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Third block
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            # Fourth block
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),

            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_lstm_model(self, input_shape, num_classes):
        """Create LSTM model for sequential pose data"""
        # Reshape input for LSTM (samples, timesteps, features)
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Reshape((33, 4)),  # 33 landmarks, 4 coordinates each

            # LSTM layers
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_cnn_model(self, input_shape, num_classes):
        """Create 1D CNN model"""
        model = models.Sequential([
            layers.Input(shape=(input_shape, 1)),

            # First conv block
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Second conv block
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),

            # Third conv block
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),

            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def create_hybrid_model(self, input_shape, num_classes):
        """Create hybrid CNN-LSTM model"""
        model = models.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Reshape((33, 4)),  # 33 landmarks, 4 coordinates

            # Conv layers
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),

            # LSTM layers
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),

            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])

        return model

    def compile_model(self, model, learning_rate=0.001):
        """Compile model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   epochs=100, batch_size=32, model_name='dl_model'):
        """Train deep learning model"""

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'models/{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # Train
        print(f"\n🚀 Training {model_name}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model on test set"""
        results = model.evaluate(X_test, y_test, verbose=0)

        print(f"\n📊 Test Results:")
        print(f"   Loss: {results[0]:.4f}")
        print(f"   Accuracy: {results[1]:.4f}")
        print(f"   Top-5 Accuracy: {results[2]:.4f}")

        return {
            'loss': results[0],
            'accuracy': results[1],
            'top5_accuracy': results[2]
        }

    def train_all_models(self):
        """Train all DL models"""
        # Load data
        print("📥 Loading data...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes = self.load_data()

        input_shape = X_train.shape[1]

        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Input shape: {input_shape}")
        print(f"   Number of classes: {num_classes}")

        results = {}

        # Train Dense NN
        print("\n" + "=" * 60)
        print("🧠 Training Dense Neural Network")
        print("=" * 60)
        model_dnn = self.create_dense_nn_model(input_shape, num_classes)
        model_dnn = self.compile_model(model_dnn)
        print(model_dnn.summary())
        history_dnn = self.train_model(
            model_dnn, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, model_name='dense_nn'
        )
        results['Dense_NN'] = self.evaluate_model(model_dnn, X_test, y_test)

        # Train CNN
        print("\n" + "=" * 60)
        print("🌐 Training 1D CNN")
        print("=" * 60)
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        model_cnn = self.create_cnn_model(input_shape, num_classes)
        model_cnn = self.compile_model(model_cnn)
        print(model_cnn.summary())
        history_cnn = self.train_model(
            model_cnn, X_train_cnn, y_train, X_val_cnn, y_val,
            epochs=100, batch_size=32, model_name='cnn_1d'
        )
        results['CNN_1D'] = self.evaluate_model(model_cnn, X_test_cnn, y_test)

        # Train LSTM
        print("\n" + "=" * 60)
        print("🔄 Training LSTM")
        print("=" * 60)
        model_lstm = self.create_lstm_model(input_shape, num_classes)
        model_lstm = self.compile_model(model_lstm)
        print(model_lstm.summary())
        history_lstm = self.train_model(
            model_lstm, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, model_name='lstm'
        )
        results['LSTM'] = self.evaluate_model(model_lstm, X_test, y_test)

        # Train Hybrid
        print("\n" + "=" * 60)
        print("⚡ Training Hybrid CNN-LSTM")
        print("=" * 60)
        model_hybrid = self.create_hybrid_model(input_shape, num_classes)
        model_hybrid = self.compile_model(model_hybrid)
        print(model_hybrid.summary())
        history_hybrid = self.train_model(
            model_hybrid, X_train, y_train, X_val, y_val,
            epochs=100, batch_size=32, model_name='hybrid_cnn_lstm'
        )
        results['Hybrid'] = self.evaluate_model(model_hybrid, X_test, y_test)

        # Find best model
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']

        print("\n" + "=" * 60)
        print("🏆 Best Deep Learning Model")
        print("=" * 60)
        print(f"   Model: {best_model_name}")
        print(f"   Accuracy: {best_accuracy:.4f}")

        # Save results
        with open('models/dl_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

if __name__ == "__main__":
    trainer = DLTrainer()

    # Check if processed data exists
    if not os.path.exists('dataset/processed/X_train.npy'):
        print("❌ Processed data not found!")
        print("   Please run data_preprocessor.py first")
    else:
        results = trainer.train_all_models()
        print("\n✅ Deep Learning Training Complete!")
