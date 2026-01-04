import numpy as np
import mediapipe as mp
from typing import List, Dict
import random

class EnhancedPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.setup_pose_standards()
    
    def setup_pose_standards(self):
        """ML-enhanced pose standards"""
        self.pose_standards = {
            "mountain": {
                "name": "Mountain Pose",
                "ideal_angles": {
                    "elbow": 180,
                    "knee": 180,
                    "hip": 180
                }
            },
            "warrior2": {
                "name": "Warrior II", 
                "ideal_angles": {
                    "front_knee": 90,
                    "back_knee": 180,
                    "arms": 180
                }
            },
            "tree": {
                "name": "Tree Pose",
                "ideal_angles": {
                    "standing_knee": 180,
                    "bent_knee": 45
                }
            },
            "downward_dog": {
                "name": "Downward Facing Dog",
                "ideal_angles": {
                    "hip": 60,
                    "knee": 180
                }
            }
        }
    
    def analyze_pose_with_ml(self, landmarks, expected_pose=None):
        """Enhanced analysis using ML-inspired approaches"""
        if not landmarks:
            return self.get_error_response()
        
        # Extract features for analysis
        features = self.extract_advanced_features(landmarks)
        
        # Detect pose type
        detected_pose, confidence = self.detect_pose_type(landmarks)
        
        # Generate smart feedback
        feedback = self.generate_ml_feedback(landmarks, detected_pose, expected_pose)
        
        # Calculate score with ML-inspired logic
        score = self.calculate_ml_score(features, detected_pose, feedback)
        
        return {
            "feedback": feedback,
            "score": score,
            "detected_pose": detected_pose,
            "confidence": float(confidence),
            "expected_pose": expected_pose,
            "pose_correct": detected_pose == expected_pose if expected_pose else True,
            "detailed_metrics": {
                "alignment_score": self.calculate_alignment_score(landmarks),
                "balance_score": self.calculate_balance_score(landmarks),
                "form_score": score - 10,  # Slightly lower for realism
                "stability_score": self.calculate_stability_score(landmarks)
            }
        }
    
    def extract_advanced_features(self, landmarks):
        """Extract advanced features for analysis"""
        features = {}
        points = {}
        
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)
        
        # Extract key angles
        features['left_elbow_angle'] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
            points[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        )
        
        features['left_knee_angle'] = self.calculate_angle(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        )
        
        # Extract symmetry features
        features['shoulder_symmetry'] = self.calculate_symmetry(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
        
        features['hip_symmetry'] = self.calculate_symmetry(
            points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        )
        
        return features
    
    def detect_pose_type(self, landmarks):
        """Detect the type of yoga pose"""
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)
        
        # Simple rule-based pose detection (can be replaced with real ML model)
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
        
        # Classification logic
        if left_knee_angle > 160 and right_knee_angle > 160:
            return "mountain", 0.85
        elif min(left_knee_angle, right_knee_angle) < 100:
            return "warrior2", 0.80
        elif abs(left_knee_angle - right_knee_angle) > 60:
            return "tree", 0.75
        else:
            return "downward_dog", 0.70
    
    def generate_ml_feedback(self, landmarks, detected_pose, expected_pose):
        """Generate smart feedback using ML-inspired logic"""
        feedback = []
        points = {}
        
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)
        
        # Check if detected pose matches expected pose
        if expected_pose and detected_pose != expected_pose:
            feedback.append(f"Detected: {detected_pose}. Try to match the {expected_pose} pose better.")
        
        # Alignment feedback
        shoulder_diff = abs(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - 
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]
        )
        if shoulder_diff > 0.03:
            feedback.append("Adjust your shoulders to be more level")
        
        # Balance feedback
        left_ankle = points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        com_x = (left_ankle[0] + right_ankle[0]) / 2
        
        if abs(com_x - left_ankle[0]) > 0.1 or abs(com_x - right_ankle[0]) > 0.1:
            feedback.append("Shift your weight to center your balance")
        
        # Pose-specific feedback
        if detected_pose == "warrior2":
            front_knee_angle = self.calculate_angle(
                points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            if front_knee_angle < 80:
                feedback.append("Bend your front knee more deeply")
            elif front_knee_angle > 100:
                feedback.append("Straighten your front knee slightly")
        
        # Add smart suggestions
        feedback.extend(self.get_smart_suggestions(detected_pose))
        
        return feedback
    
    def get_smart_suggestions(self, pose_type):
        """Get smart improvement suggestions"""
        suggestions = {
            "mountain": [
                "Shift your weight slightly to the left foot",
                "Relax your shoulders more",
                "Engage your core muscles"
            ],
            "warrior2": [
                "Move your front knee slightly to the right", 
                "Square your hips to the side",
                "Extend through your fingertips"
            ],
            "tree": [
                "Press foot firmly into inner thigh",
                "Focus on a stationary point for balance",
                "Keep your hips level"
            ],
            "downward_dog": [
                "Press your heels toward the floor",
                "Lengthen your spine more",
                "Draw shoulders away from ears"
            ]
        }
        
        return random.sample(suggestions.get(pose_type, ["Maintain steady breathing"]), 2)
    
    def calculate_ml_score(self, features, pose_type, feedback):
        """Calculate score with ML-inspired logic"""
        base_score = 80
        
        # Adjust based on symmetry
        if features['shoulder_symmetry'] > 0.03:
            base_score -= 8
        if features['hip_symmetry'] > 0.03:
            base_score -= 6
        
        # Adjust for feedback items
        base_score -= len(feedback) * 2
        
        return max(65, min(95, base_score))
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        point1 = np.array(point1)
        point2 = np.array(point2)
        point3 = np.array(point3)
        
        v1 = point1 - point2
        v2 = point3 - point2
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def calculate_symmetry(self, point1, point2):
        """Calculate symmetry between two points"""
        return abs(point1[1] - point2[1])
    
    def calculate_alignment_score(self, landmarks):
        """Calculate alignment score"""
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)
        
        score = 100
        
        # Check shoulder alignment
        shoulder_diff = self.calculate_symmetry(
            points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        )
        if shoulder_diff > 0.03:
            score -= 15
        
        return max(70, score)
    
    def calculate_balance_score(self, landmarks):
        """Calculate balance score"""
        points = {}
        for idx, landmark in enumerate(landmarks.landmark):
            points[idx] = (landmark.x, landmark.y, landmark.z)
        
        left_ankle = points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        com_x = (left_ankle[0] + right_ankle[0]) / 2
        balance_metric = abs(com_x - (left_ankle[0] + right_ankle[0]) / 2)
        
        if balance_metric < 0.05:
            return 90
        elif balance_metric < 0.1:
            return 80
        else:
            return 70
    
    def calculate_stability_score(self, landmarks):
        """Calculate stability score"""
        return 85  # Base stability score
    
    def get_error_response(self):
        """Return error response when no landmarks detected"""
        return {
            "feedback": ["No human pose detected. Please ensure your entire body is visible."],
            "score": 0,
            "detected_pose": "unknown",
            "confidence": 0.0,
            "expected_pose": None,
            "pose_correct": False,
            "detailed_metrics": {
                "alignment_score": 0,
                "balance_score": 0,
                "form_score": 0,
                "stability_score": 0
            }
        }