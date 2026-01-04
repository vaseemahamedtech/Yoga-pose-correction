import numpy as np
import mediapipe as mp
from typing import List, Dict
import random


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.setup_pose_standards()

    def setup_pose_standards(self):
        self.pose_standards = {
            "mountain": {
                "name": "Mountain Pose",
                "feedback": [
                    "Stand with feet together and arms at sides",
                    "Keep shoulders relaxed and down",
                    "Engage your thigh muscles",
                    "Lengthen your spine",
                    "Look straight ahead"
                ]
            },
            "warrior2": {
                "name": "Warrior II",
                "feedback": [
                    "Front knee should be at 90 degrees",
                    "Back leg should be straight",
                    "Hips should face the side",
                    "Arms should be parallel to floor",
                    "Gaze over front fingertips"
                ]
            },
            "tree": {
                "name": "Tree Pose",
                "feedback": [
                    "Place foot on inner thigh or calf (not knee)",
                    "Standing leg should be straight",
                    "Hips should be level",
                    "Bring hands to prayer position",
                    "Find a focal point for balance"
                ]
            },
            "downward_dog": {
                "name": "Downward Facing Dog",
                "feedback": [
                    "Form an inverted V shape with body",
                    "Hands shoulder-width apart",
                    "Feet hip-width apart",
                    "Heels reaching toward floor",
                    "Lengthen your spine"
                ]
            }
        }

    def generate_smart_feedback(self, landmarks, pose_type: str) -> Dict:
        if not landmarks:
            return {
                "feedback": ["No human pose detected. Please ensure your entire body is visible in the frame."],
                "score": 0,
                "pose_type": pose_type,
                "corrections": []
            }
        points = {idx: (lm.x, lm.y, lm.z) for idx, lm in enumerate(landmarks.landmark)}
        feedback = []
        score = 80
        alignment_feedback = self.check_basic_alignment(points)
        feedback.extend(alignment_feedback)
        pose_feedback = self.check_pose_specific(points, pose_type)
        feedback.extend(pose_feedback)
        specific_corrections = self.generate_specific_corrections(pose_type)
        feedback.extend(specific_corrections)
        score = max(60, score - len(feedback) * 3)
        return {
            "feedback": feedback,
            "score": score,
            "pose_type": pose_type,
            "pose_name": self.pose_standards.get(pose_type, {}).get("name", "Yoga Pose"),
            "corrections": specific_corrections,
            "detailed_metrics": {
                "alignment_score": 85 - len(alignment_feedback) * 5,
                "balance_score": 80 - len(specific_corrections) * 4,
                "form_score": 75
            }
        }

    def check_basic_alignment(self, points: Dict) -> List[str]:
        feedback = []
        try:
            if self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_SHOULDER) and \
               self.has_landmark(points, self.mp_pose.PoseLandmark.RIGHT_SHOULDER):
                left = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right = points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                if abs(left[1] - right[1]) > 0.03:
                    feedback.append("Keep your shoulders level and relaxed")
            if self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_HIP) and \
               self.has_landmark(points, self.mp_pose.PoseLandmark.RIGHT_HIP):
                left = points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right = points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                if abs(left[1] - right[1]) > 0.03:
                    feedback.append("Align your hips evenly")
            if self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_SHOULDER) and \
               self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_HIP) and \
               self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_ANKLE):
                shoulder = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                hip = points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                ankle = points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                if abs(shoulder[0] - hip[0]) > 0.1:
                    feedback.append("Keep your spine straight and aligned")
        except Exception as e:
            print(f"Alignment check error: {e}")
        return feedback

    def check_pose_specific(self, points: Dict, pose_type: str) -> List[str]:
        feedback = []
        try:
            if pose_type == "mountain":
                feedback.extend(self.analyze_mountain_pose(points))
            elif pose_type == "warrior2":
                feedback.extend(self.analyze_warrior2_pose(points))
            elif pose_type == "tree":
                feedback.extend(self.analyze_tree_pose(points))
            elif pose_type == "downward_dog":
                feedback.extend(self.analyze_downward_dog_pose(points))
        except Exception as e:
            print(f"Pose-specific check error: {e}")
        return feedback

    def analyze_mountain_pose(self, points: Dict) -> List[str]:
        feedback = []
        if (self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_ELBOW) and
            self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_SHOULDER)):
            elbow_angle = self.calculate_angle(
                points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                points[self.mp_pose.PoseLandmark.LEFT_WRIST.value] if self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_WRIST) else 180
            )
            if elbow_angle < 160:
                feedback.append("Relax your arms by your sides")
        return feedback

    def analyze_warrior2_pose(self, points: Dict) -> List[str]:
        feedback = []
        if (self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_HIP) and
            self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_KNEE) and
            self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_ANKLE)):
            knee_angle = self.calculate_angle(
                points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
            )
            if knee_angle > 120:
                feedback.append("Bend your front knee deeper toward 90 degrees")
            elif knee_angle < 80:
                feedback.append("Slightly straighten your front knee")
        return feedback

    def analyze_tree_pose(self, points: Dict) -> List[str]:
        feedback = []
        if (self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_KNEE) and
            self.has_landmark(points, self.mp_pose.PoseLandmark.RIGHT_KNEE)):
            left_knee = points[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
            right_knee = points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
            if abs(left_knee[1] - right_knee[1]) > 0.2:
                feedback.append("Bring your bent knee higher or lower for better balance")
        return feedback

    def analyze_downward_dog_pose(self, points: Dict) -> List[str]:
        feedback = []
        if (self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_SHOULDER) and
            self.has_landmark(points, self.mp_pose.PoseLandmark.LEFT_HIP)):
            shoulder = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            hip = points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            if shoulder[1] > hip[1]:
                feedback.append("Lift your hips higher toward the ceiling")
        return feedback

    def generate_specific_corrections(self, pose_type: str) -> List[str]:
        corrections_pool = {
            "mountain": [
                "Shift your weight slightly to the left foot",
                "Move your right foot slightly outward",
                "Relax your shoulders more",
                "Engage your core muscles",
                "Lengthen through the crown of your head"
            ],
            "warrior2": [
                "Move your front knee slightly to the right",
                "Adjust your back foot position",
                "Extend your arms more fully",
                "Square your hips to the side",
                "Deepen your knee bend slightly"
            ],
            "tree": [
                "Shift your weight to the standing leg",
                "Adjust your foot position on the inner thigh",
                "Bring your hands to heart center",
                "Find a focal point for better balance",
                "Engage your core for stability"
            ],
            "downward_dog": [
                "Press your heels toward the floor",
                "Lengthen your spine more",
                "Spread your fingers wide",
                "Bend your knees slightly if needed",
                "Draw your shoulders away from ears"
            ]
        }
        pose_corrections = corrections_pool.get(pose_type, [
            "Adjust your posture for better alignment",
            "Breathe deeply and relax into the pose",
            "Find your balance and stability"
        ])
        return random.sample(pose_corrections, min(3, len(pose_corrections)))

    def calculate_angle(self, point1, point2, point3):
        try:
            point1 = np.array(point1)
            point2 = np.array(point2)
            point3 = np.array(point3)
            v1 = point1 - point2
            v2 = point3 - point2
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1, 1)
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        except:
            return 180

    def has_landmark(self, points, landmark):
        return landmark.value in points and points[landmark.value] is not None
