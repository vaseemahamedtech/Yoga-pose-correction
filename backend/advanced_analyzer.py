# Advanced analyzer backend module
import numpy as np
from typing import List, Dict, Tuple
import mediapipe as mp

class AdvancedPoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.setup_pose_standards()
    
    def setup_pose_standards(self):
        self.pose_standards = {
            "mountain": {
                "joint_angles": {
                    "left_elbow": (160, 200),
                    "right_elbow": (160, 200),
                    "left_knee": (170, 190),
                    "right_knee": (170, 190)
                },
                "alignment_tolerances": {
                    "shoulders_level": 0.03,
                    "hips_level": 0.03,
                    "spine_straight": 0.05
                }
            },
            "warrior2": {
                "joint_angles": {
                    "front_knee": (85, 95),
                    "back_knee": (170, 190),
                    "arms_parallel": (170, 190)
                },
                "alignment_tolerances": {
                    "front_knee_over_ankle": 0.1,
                    "hips_square": 0.15
                }
            },
            "tree": {
                "joint_angles": {
                    "standing_knee": (170, 190),
                    "bent_knee": (30, 60)
                },
                "alignment_tolerances": {
                    "hips_level": 0.04,
                    "spine_straight": 0.06
                }
            },
            "downward_dog": {
                "joint_angles": {
                    "hip_angle": (50, 70),
                    "knee_angle": (170, 190),
                    "elbow_angle": (170, 190)
                },
                "alignment_tolerances": {
                    "spine_straight": 0.08,
                    "heels_down": 0.1
                }
            }
        }
    
    def comprehensive_analysis(self, landmarks, pose_type: str) -> Dict:
        """Perform comprehensive pose analysis with validation"""
        try:
            if not landmarks:
                return self.get_default_analysis("No pose landmarks detected")
            
            points = self.extract_key_points(landmarks)
            if not points:
                return self.get_default_analysis("No valid key points detected")
            
            # Validate that we have enough points for analysis
            if not self.has_sufficient_points(points):
                return self.get_default_analysis("Insufficient body points detected")
            
            analysis = {
                "feedback": [],
                "corrections": [],
                "score": 100,
                "detailed_metrics": {},
                "pose_type": pose_type
            }
            
            # Only analyze if we have valid pose data
            if self.is_valid_pose(points):
                # Angle-based analysis
                angle_feedback = self.analyze_joint_angles(points, pose_type)
                analysis["feedback"].extend(angle_feedback)
                
                # Alignment analysis
                alignment_feedback = self.analyze_body_alignment(points, pose_type)
                analysis["feedback"].extend(alignment_feedback)
                
                # Balance analysis
                balance_feedback = self.analyze_balance(points, pose_type)
                analysis["feedback"].extend(balance_feedback)
                
                # Pose-specific analysis
                pose_specific_feedback = self.analyze_pose_specific(points, pose_type)
                analysis["feedback"].extend(pose_specific_feedback)
                
                # Calculate final score
                analysis["score"] = self.calculate_comprehensive_score(analysis["feedback"])
                analysis["detailed_metrics"] = self.get_detailed_metrics(points, pose_type)
            else:
                analysis["feedback"].append("Pose detected but unable to analyze. Please ensure clear full-body view.")
                analysis["score"] = 50
            
            return analysis
            
        except Exception as e:
            print(f"Error in comprehensive_analysis: {e}")
            return self.get_default_analysis(f"Analysis error: {str(e)}")
    
    def get_default_analysis(self, message: str) -> Dict:
        """Return default analysis when pose detection fails"""
        return {
            "feedback": [message, "Please ensure full body is visible and well-lit"],
            "corrections": [],
            "score": 0,
            "detailed_metrics": {},
            "pose_type": "unknown"
        }
    
    def has_sufficient_points(self, points: Dict) -> bool:
        """Check if we have enough body points for analysis"""
        required_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]
        
        for point in required_points:
            if point.value not in points:
                return False
        return True
    
    def is_valid_pose(self, points: Dict) -> bool:
        """Validate if the detected points represent a reasonable human pose"""
        try:
            # Check if shoulders and hips are in reasonable positions
            left_shoulder = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_hip = points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
            right_hip = points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
            
            # Shoulders should be above hips
            if left_shoulder[1] >= left_hip[1] or right_shoulder[1] >= right_hip[1]:
                return False
            
            # Shoulders should be reasonably spaced
            shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_distance < 0.1:  # Too close together
                return False
            
            return True
            
        except Exception as e:
            print(f"Error in pose validation: {e}")
            return False
    
    def analyze_joint_angles(self, points: Dict, pose_type: str) -> List[str]:
        """Analyze joint angles against pose standards"""
        feedback = []
        
        try:
            if pose_type == "mountain":
                # Analyze elbow angles only if wrists are detected
                if (self.mp_pose.PoseLandmark.LEFT_WRIST.value in points and
                    self.mp_pose.PoseLandmark.RIGHT_WRIST.value in points):
                    
                    left_elbow_angle = self.calculate_angle(
                        points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                        points[self.mp_pose.PoseLandmark.LEFT_ELBOW.value],
                        points[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                    )
                    
                    if left_elbow_angle < 160:
                        feedback.append("Straighten your left arm more")
                    elif left_elbow_angle > 200:
                        feedback.append("Bend your left arm slightly")

            elif pose_type == "warrior2":
                # Analyze front knee angle
                if (self.mp_pose.PoseLandmark.LEFT_HIP.value in points and
                    self.mp_pose.PoseLandmark.LEFT_KNEE.value in points and
                    self.mp_pose.PoseLandmark.LEFT_ANKLE.value in points):
                    
                    front_knee_angle = self.calculate_angle(
                        points[self.mp_pose.PoseLandmark.LEFT_HIP.value],
                        points[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
                        points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    )
                    
                    if front_knee_angle < 85:
                        feedback.append("Bend your front knee more - aim for 90 degrees")
                    elif front_knee_angle > 95:
                        feedback.append("Reduce bend in your front knee - aim for 90 degrees")

            elif pose_type == "tree":
                # Analyze standing leg
                if (self.mp_pose.PoseLandmark.RIGHT_HIP.value in points and
                    self.mp_pose.PoseLandmark.RIGHT_KNEE.value in points and
                    self.mp_pose.PoseLandmark.RIGHT_ANKLE.value in points):
                    
                    standing_knee_angle = self.calculate_angle(
                        points[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
                        points[self.mp_pose.PoseLandmark.RIGHT_KNEE.value],
                        points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                    )
                    
                    if standing_knee_angle < 170:
                        feedback.append("Straighten your standing leg more")

        except Exception as e:
            print(f"Error in joint angle analysis: {e}")
            
        return feedback
    
    def analyze_body_alignment(self, points: Dict, pose_type: str) -> List[str]:
        """Analyze body alignment and symmetry"""
        feedback = []
        
        try:
            # Shoulder alignment
            if (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value in points and 
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value in points):
                left_shoulder = points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1])
                
                if shoulder_diff > 0.03:
                    if left_shoulder[1] > right_shoulder[1]:
                        feedback.append("Lower your left shoulder slightly")
                    else:
                        feedback.append("Lower your right shoulder slightly")
            
            # Hip alignment
            if (self.mp_pose.PoseLandmark.LEFT_HIP.value in points and 
                self.mp_pose.PoseLandmark.RIGHT_HIP.value in points):
                left_hip = points[self.mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = points[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
                hip_diff = abs(left_hip[1] - right_hip[1])
                
                if hip_diff > 0.03:
                    if left_hip[1] > right_hip[1]:
                        feedback.append("Shift weight to your right leg slightly")
                    else:
                        feedback.append("Shift weight to your left leg slightly")
                        
        except Exception as e:
            print(f"Error in body alignment analysis: {e}")
            
        return feedback
    
    def analyze_balance(self, points: Dict, pose_type: str) -> List[str]:
        """Analyze balance and weight distribution"""
        feedback = []
        
        try:
            if (self.mp_pose.PoseLandmark.LEFT_ANKLE.value in points and 
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value in points):
                left_ankle = points[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
                right_ankle = points[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
                
                if pose_type == "tree":
                    # For tree pose, check if COM is over standing foot
                    standing_ankle = right_ankle
                    com_x = (left_ankle[0] + right_ankle[0]) / 2
                    com_offset = abs(com_x - standing_ankle[0])
                    
                    if com_offset > 0.05:
                        if com_x > standing_ankle[0]:
                            feedback.append("Shift your weight slightly to the left")
                        else:
                            feedback.append("Shift your weight slightly to the right")
                            
        except Exception as e:
            print(f"Error in balance analysis: {e}")
            
        return feedback
    
    def analyze_pose_specific(self, points: Dict, pose_type: str) -> List[str]:
        """Pose-specific analysis and corrections"""
        feedback = []
        
        try:
            if pose_type == "warrior2":
                if (self.mp_pose.PoseLandmark.LEFT_WRIST.value in points and
                    self.mp_pose.PoseLandmark.RIGHT_WRIST.value in points):
                    
                    left_wrist = points[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                    right_wrist = points[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                    
                    # Check if arms are parallel to floor
                    arms_level_diff = abs(left_wrist[1] - right_wrist[1])
                    if arms_level_diff > 0.05:
                        feedback.append("Keep both arms at the same height - parallel to the floor")
                        
        except Exception as e:
            print(f"Error in pose-specific analysis: {e}")
            
        return feedback
    
    def extract_key_points(self, landmarks):
        """Extract and normalize key points with visibility check"""
        points = {}
        try:
            for landmark in self.mp_pose.PoseLandmark:
                if (landmarks.landmark[landmark.value] and 
                    landmarks.landmark[landmark.value].visibility > 0.1):  # Only use visible points
                    points[landmark.value] = (
                        landmarks.landmark[landmark.value].x,
                        landmarks.landmark[landmark.value].y,
                        landmarks.landmark[landmark.value].z
                    )
        except Exception as e:
            print(f"Error extracting key points: {e}")
        return points
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points in 3D space"""
        try:
            point1 = np.array(point1)
            point2 = np.array(point2)
            point3 = np.array(point3)
            
            # Vectors from point2
            v1 = point1 - point2
            v2 = point3 - point2
            
            # Calculate angle
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cosine_angle = np.clip(cosine_angle, -1, 1)  # Ensure within valid range
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 180  # Return default straight angle
    
    def calculate_comprehensive_score(self, feedback: List[str]) -> int:
        """Calculate pose score based on feedback items"""
        base_score = 100
        penalty = min(len(feedback) * 8, 60)  # Max penalty of 60
        return max(base_score - penalty, 0)  # Minimum score of 0
    
    def get_detailed_metrics(self, points: Dict, pose_type: str) -> Dict:
        """Get detailed metrics for the pose"""
        metrics = {}
        
        try:
            # Calculate various angles and measurements
            if (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value in points and 
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value in points):
                metrics["shoulder_alignment"] = abs(
                    points[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] - 
                    points[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]
                )
            
            if (self.mp_pose.PoseLandmark.LEFT_HIP.value in points and 
                self.mp_pose.PoseLandmark.RIGHT_HIP.value in points):
                metrics["hip_alignment"] = abs(
                    points[self.mp_pose.PoseLandmark.LEFT_HIP.value][1] - 
                    points[self.mp_pose.PoseLandmark.RIGHT_HIP.value][1]
                )
        except Exception as e:
            print(f"Error getting detailed metrics: {e}")
            
        return metrics