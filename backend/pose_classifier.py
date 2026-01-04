import tensorflow as tf
from tensorflow import keras
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
import joblib


class PoseClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.mp_pose = mp.solutions.pose
        self.load_or_create_model()

    def load_or_create_model(self):
        try:
            self.model = tf.keras.models.load_model('models/pose_classifier.h5')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            print("✅ Pre-trained model loaded")
        except:
            print("🔄 Creating new model architecture...")
            self.create_model()

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(99,)),  # 33 landmarks * 3 coordinates
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(4, activation='softmax')  # 4 yoga poses
        ])

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Initialize with basic labels
        self.label_encoder.fit(['mountain', 'warrior2', 'tree', 'downward_dog'])

    def extract_features(self, landmarks):
        features = []
        for landmark in landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features).reshape(1, -1)

    def predict_pose(self, landmarks):
        if landmarks is None:
            return "unknown", 0.0

        features = self.extract_features(landmarks)

        # If model is untrained, use rule-based fallback
        if len(self.model.get_weights()) == 0:
            return self.rule_based_classification(landmarks)

        predictions = self.model.predict(features, verbose=0)
        confidence = np.max(predictions)
        predicted_class = self.label_encoder.inverse_transform([np.argmax(predictions)])[0]

        return predicted_class, confidence

    def rule_based_classification(self, landmarks):
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)

        # Simple rule-based classification
        left_knee_angle = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )

        right_knee_angle = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )

        # Classification rules
        if left_knee_angle > 160 and right_knee_angle > 160:
            return "mountain", 0.8
        elif min(left_knee_angle, right_knee_angle) < 100:
            return "warrior2", 0.7
        elif abs(left_knee_angle - right_knee_angle) > 60:
            return "tree", 0.75
        else:
            return "downward_dog", 0.6

    def calculate_angle(self, point1, point2, point3):
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)

        v1 = point1 - point2
        v2 = point3 - point2

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def train_model(self, X_train, y_train, epochs=100):
        y_encoded = self.label_encoder.transform(y_train)
        y_categorical = keras.utils.to_categorical(y_encoded, 4)

        history = self.model.fit(
            X_train, y_categorical,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        self.model.save('models/pose_classifier.h5')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')

        return history
