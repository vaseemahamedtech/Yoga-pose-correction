"""
Data Preprocessing Module for Yoga Pose Dataset
Handles dataset retrieval, preprocessing, augmentation, and feature engineering
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
from pathlib import Path
import requests
from tqdm import tqdm

class YogaDataPreprocessor:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Yoga pose classes
        self.pose_classes = ['mountain', 'warrior2', 'tree', 'downward_dog', 
                            'cobra', 'plank', 'child', 'triangle']

    def download_dataset_instructions(self):
        """Provide instructions for dataset download"""
        instructions = """
        📥 YOGA DATASET DOWNLOAD INSTRUCTIONS
        =====================================

        This project supports multiple yoga datasets:

        1. Yoga-82 Dataset (Recommended - 82 poses)
           - Visit: https://sites.google.com/view/yoga-82/home
           - Fill the form to get download link
           - Download and extract to: dataset/raw/

        2. Yoga-16 Dataset (Kaggle - 16 poses)
           - Visit: https://www.kaggle.com/datasets/mohiuddin2531/yoga-16
           - Download and extract to: dataset/raw/

        3. Yoga Poses Dataset (Kaggle - 5 poses)
           - Visit: https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification
           - Download and extract to: dataset/raw/

        After downloading, run:
        python backend/data_preprocessor.py --preprocess

        This will:
        - Extract pose landmarks using MediaPipe
        - Create feature vectors for ML models
        - Split into train/validation/test sets
        - Save processed data for training
        """
        return instructions

    def extract_landmarks_from_image(self, image_path):
        """Extract pose landmarks from image using MediaPipe"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks)
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def process_dataset_folder(self, raw_data_path):
        """Process all images in dataset folder"""
        features = []
        labels = []

        raw_path = Path(raw_data_path)
        if not raw_path.exists():
            print(f"❌ Dataset path {raw_data_path} not found!")
            print(self.download_dataset_instructions())
            return None, None

        # Find all pose folders
        pose_folders = [f for f in raw_path.iterdir() if f.is_dir()]

        print(f"\n🔍 Found {len(pose_folders)} pose categories")
        print("Processing images...")

        for pose_folder in tqdm(pose_folders, desc="Processing poses"):
            pose_name = pose_folder.name.lower()

            # Get all images in folder
            image_files = list(pose_folder.glob('*.jpg')) + \
                         list(pose_folder.glob('*.png')) + \
                         list(pose_folder.glob('*.jpeg'))

            for img_path in tqdm(image_files, desc=f"  {pose_name}", leave=False):
                landmarks = self.extract_landmarks_from_image(img_path)
                if landmarks is not None:
                    features.append(landmarks)
                    labels.append(pose_name)

        return np.array(features), np.array(labels)

    def create_train_test_split(self, features, labels, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Second split: train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def normalize_features(self, X_train, X_val, X_test):
        """Normalize features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def save_processed_data(self, data_splits, output_dir='dataset/processed'):
        """Save processed data"""
        os.makedirs(output_dir, exist_ok=True)

        # Save numpy arrays
        np.save(f'{output_dir}/X_train.npy', data_splits['X_train'])
        np.save(f'{output_dir}/y_train.npy', data_splits['y_train'])
        np.save(f'{output_dir}/X_val.npy', data_splits['X_val'])
        np.save(f'{output_dir}/y_val.npy', data_splits['y_val'])
        np.save(f'{output_dir}/X_test.npy', data_splits['X_test'])
        np.save(f'{output_dir}/y_test.npy', data_splits['y_test'])

        print(f"✅ Processed data saved to {output_dir}")
        print(f"   Training samples: {len(data_splits['X_train'])}")
        print(f"   Validation samples: {len(data_splits['X_val'])}")
        print(f"   Test samples: {len(data_splits['X_test'])}")

if __name__ == "__main__":
    preprocessor = YogaDataPreprocessor()

    # Show download instructions
    print(preprocessor.download_dataset_instructions())

    # Check if raw data exists
    raw_path = 'dataset/raw'
    if not os.path.exists(raw_path) or len(os.listdir(raw_path)) <= 1:
        print("\n⚠️  Please download dataset first following above instructions!")
    else:
        print("\n🚀 Starting data preprocessing...")
        features, labels = preprocessor.process_dataset_folder(raw_path)

        if features is not None:
            print(f"\n✅ Extracted features from {len(features)} images")

            # Create splits
            data_splits = preprocessor.create_train_test_split(features, labels)

            # Normalize
            data_splits['X_train'], data_splits['X_val'], data_splits['X_test'] = \
                preprocessor.normalize_features(
                    data_splits['X_train'], 
                    data_splits['X_val'], 
                    data_splits['X_test']
                )

            # Save
            preprocessor.save_processed_data(data_splits)
