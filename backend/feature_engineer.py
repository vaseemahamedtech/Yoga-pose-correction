"""
Feature Engineering Module for Yoga Pose Classification
Extracts advanced features from pose landmarks
"""

import numpy as np
import mediapipe as mp

class YogaFeatureEngineer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)

        v1 = point1 - point2
        v2 = point3 - point2

        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def extract_angle_features(self, landmarks):
        """Extract angle-based features from landmarks"""
        # Landmarks are in format [x, y, z, visibility] * 33
        # Reshape to get coordinates
        landmarks_3d = landmarks.reshape(-1, 4)[:, :3]

        features = []

        # Key angles for yoga poses
        # Left elbow
        left_elbow_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        features.append(left_elbow_angle)

        # Right elbow
        right_elbow_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        )
        features.append(right_elbow_angle)

        # Left knee
        left_knee_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        features.append(left_knee_angle)

        # Right knee
        right_knee_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        )
        features.append(right_knee_angle)

        # Hip angle
        hip_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        )
        features.append(hip_angle)

        # Shoulder angle
        shoulder_angle = self.calculate_angle(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        features.append(shoulder_angle)

        return np.array(features)

    def extract_distance_features(self, landmarks):
        """Extract distance-based features"""
        landmarks_3d = landmarks.reshape(-1, 4)[:, :3]

        features = []

        # Shoulder width
        shoulder_width = self.calculate_distance(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
        features.append(shoulder_width)

        # Hip width
        hip_width = self.calculate_distance(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        features.append(hip_width)

        # Torso length (shoulder to hip)
        torso_length = self.calculate_distance(
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        )
        features.append(torso_length)

        return np.array(features)

    def extract_symmetry_features(self, landmarks):
        """Extract symmetry features"""
        landmarks_3d = landmarks.reshape(-1, 4)[:, :3]

        features = []

        # Shoulder symmetry
        left_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        shoulder_symmetry = abs(left_shoulder[1] - right_shoulder[1])
        features.append(shoulder_symmetry)

        # Hip symmetry
        left_hip = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        hip_symmetry = abs(left_hip[1] - right_hip[1])
        features.append(hip_symmetry)

        # Knee symmetry
        left_knee = landmarks_3d[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
        right_knee = landmarks_3d[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        knee_symmetry = abs(left_knee[1] - right_knee[1])
        features.append(knee_symmetry)

        return np.array(features)

    def extract_all_features(self, landmarks):
        """Extract all engineered features"""
        # Original landmarks (132 features)
        original_features = landmarks

        # Angle features (6 features)
        angle_features = self.extract_angle_features(landmarks)

        # Distance features (3 features)
        distance_features = self.extract_distance_features(landmarks)

        # Symmetry features (3 features)
        symmetry_features = self.extract_symmetry_features(landmarks)

        # Combine all features (132 + 6 + 3 + 3 = 144 features)
        all_features = np.concatenate([
            original_features,
            angle_features,
            distance_features,
            symmetry_features
        ])

        return all_features

    def batch_extract_features(self, landmarks_batch):
        """Extract features for batch of landmarks"""
        return np.array([self.extract_all_features(lm) for lm in landmarks_batch])
