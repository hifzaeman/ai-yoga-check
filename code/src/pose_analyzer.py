"""
Pose Analysis Module for AI Yoga Instructor

This module provides the core pose analysis functionality including angle calculations,
pose stability tracking, and similarity scoring for yoga pose comparison.
"""

import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Optional, Tuple, Any


class PoseAnalyzer:
    """
    Core pose analysis class that handles pose detection, angle calculation,
    and pose similarity scoring.
    """
    
    def __init__(self):
        """Initialize the pose analyzer with MediaPipe and configuration settings."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pose stability tracking
        self.pose_history = deque(maxlen=30)
        self.stability_threshold = 15.0
        self.min_stable_frames = 20
        
        # Target pose storage
        self.target_pose = None
        self.target_angles = None
        
        # Angle tolerances for different body parts
        self.angle_tolerances = {
            'right_arm': 35.0,
            'left_arm': 35.0,
            'right_knee': 45.0,
            'left_knee': 45.0,
            'right_shoulder': 25.0,
            'left_shoulder': 25.0
        }
        
        self.pose_type_detected = None
        
    def calculate_angle(self, point1: Any, point2: Any, point3: Any) -> Optional[float]:
        """
        Calculate angle between three points with improved stability.
        
        Args:
            point1, point2, point3: MediaPipe landmark points
            
        Returns:
            Angle in degrees or None if calculation not possible
        """
        if not all([point1.visibility > 0.5, point2.visibility > 0.5, point3.visibility > 0.5]):
            return None
            
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])  # vertex point
        c = np.array([point3.x, point3.y])
        
        ba = a - b
        bc = c - b
        
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm < 1e-6 or bc_norm < 1e-6:
            return None
            
        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def extract_key_angles(self, landmarks: List[Any]) -> Optional[Dict[str, Optional[float]]]:
        """
        Extract key body angles from pose landmarks.
        
        Args:
            landmarks: List of MediaPipe pose landmarks
            
        Returns:
            Dictionary of angle names to angle values
        """
        if not landmarks:
            return None
            
        angles = {}
        
        # Right arm angle (wrist -> elbow -> shoulder)
        angles['right_arm'] = self.calculate_angle(
            landmarks[16], landmarks[14], landmarks[12]
        )
        
        # Left arm angle (wrist -> elbow -> shoulder)  
        angles['left_arm'] = self.calculate_angle(
            landmarks[15], landmarks[13], landmarks[11]
        )
        
        # Right knee angle (hip -> knee -> ankle)
        angles['right_knee'] = self.calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]
        )
        
        # Left knee angle (hip -> knee -> ankle)
        angles['left_knee'] = self.calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]
        )
        
        # Right shoulder angle (elbow -> shoulder -> hip)
        angles['right_shoulder'] = self.calculate_angle(
            landmarks[14], landmarks[12], landmarks[24]
        )
        
        # Left shoulder angle (elbow -> shoulder -> hip)
        angles['left_shoulder'] = self.calculate_angle(
            landmarks[13], landmarks[11], landmarks[23]
        )
        
        self.detect_pose_type(angles, landmarks)
        
        return angles
    
    def detect_pose_type(self, angles: Dict[str, Optional[float]], landmarks: List[Any]) -> None:
        """
        Detect what type of pose is being performed for contextual adjustments.
        
        Args:
            angles: Dictionary of angle measurements
            landmarks: List of MediaPipe pose landmarks
        """
        if not angles or not landmarks:
            return
        
        # Check if it's a balance pose (one leg lifted)
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate relative height of ankles
        left_ankle_height = left_ankle.y
        right_ankle_height = right_ankle.y
        
        height_diff = abs(left_ankle_height - right_ankle_height)
        
        if height_diff > 0.15:  # One foot significantly higher
            self.pose_type_detected = "balance_pose"
            # For balance poses, make leg angles much more lenient
            self.angle_tolerances['right_knee'] = 60.0
            self.angle_tolerances['left_knee'] = 60.0
            return
        
        # Default pose type
        self.pose_type_detected = "standard_pose"
        self.angle_tolerances['right_knee'] = 45.0
        self.angle_tolerances['left_knee'] = 45.0
    
    def is_pose_stable(self, current_angles: Dict[str, Optional[float]]) -> bool:
        """
        Check if pose has been stable for enough frames.
        
        Args:
            current_angles: Current angle measurements
            
        Returns:
            True if pose is stable, False otherwise
        """
        if not current_angles:
            return False
            
        self.pose_history.append(current_angles)
        
        if len(self.pose_history) < self.min_stable_frames:
            return False
        
        recent_poses = list(self.pose_history)[-self.min_stable_frames:]
        stable_angles = 0
        total_angles = 0
        
        for angle_name in current_angles:
            if current_angles[angle_name] is None:
                continue
                
            angle_values = [pose[angle_name] for pose in recent_poses 
                           if pose[angle_name] is not None]
            
            if len(angle_values) < self.min_stable_frames * 0.6:
                continue
                
            total_angles += 1
            angle_std = np.std(angle_values)
            
            # Use adaptive threshold based on pose type
            threshold = self.stability_threshold
            if self.pose_type_detected == "balance_pose":
                threshold *= 1.5
                
            if angle_std <= threshold:
                stable_angles += 1
        
        stability_ratio = stable_angles / max(total_angles, 1)
        return stability_ratio >= 0.7
    
    def calculate_pose_similarity(self, user_angles: Dict[str, Optional[float]]) -> float:
        """
        Calculate overall pose similarity with weighted scoring.
        
        Args:
            user_angles: User's current angle measurements
            
        Returns:
            Similarity score from 0-100
        """
        if not self.target_angles or not user_angles:
            return 0
        
        total_weighted_score = 0
        total_weights = 0
        
        # Weight different angles by importance
        angle_weights = {
            'right_arm': 1.2,
            'left_arm': 1.2,
            'right_knee': 1.5,
            'left_knee': 1.5,
            'right_shoulder': 1.0,
            'left_shoulder': 1.0
        }
        
        for angle_name in self.target_angles:
            target_angle = self.target_angles[angle_name]
            user_angle = user_angles[angle_name]
            
            if target_angle is None or user_angle is None:
                continue
            
            weight = angle_weights.get(angle_name, 1.0)
            tolerance = self.angle_tolerances.get(angle_name, 30.0)
            
            # Use adaptive tolerance for balance poses
            if self.pose_type_detected == "balance_pose" and "knee" in angle_name:
                tolerance *= 1.5
            
            difference = abs(target_angle - user_angle)
            
            # Non-linear scoring that's more forgiving
            if difference <= tolerance * 0.5:  # Very close
                angle_score = 100
            elif difference <= tolerance:  # Within tolerance
                ratio = (difference - tolerance * 0.5) / (tolerance * 0.5)
                angle_score = 100 - (ratio * 15)
            else:  # Outside tolerance
                excess = difference - tolerance
                max_excess = 90
                penalty_ratio = min(excess / max_excess, 1.0)
                angle_score = 85 - (penalty_ratio * 60)
            
            angle_score = max(angle_score, 25)  # Minimum score of 25%
            
            total_weighted_score += angle_score * weight
            total_weights += weight
        
        final_score = total_weighted_score / max(total_weights, 1)
        
        # Apply overall pose bonus for good matches
        if final_score > 90:
            final_score = min(final_score + 5, 98)
        elif final_score > 80:
            final_score = min(final_score + 2, 95)
        
        return final_score
    
    def generate_feedback(self, user_angles: Dict[str, Optional[float]]) -> List[str]:
        """
        Generate specific feedback based on pose comparison.
        
        Args:
            user_angles: User's current angle measurements
            
        Returns:
            List of feedback instructions
        """
        if not self.target_angles or not user_angles:
            return []
        
        feedback = []
        minor_issues = []
        
        for angle_name in self.target_angles:
            target_angle = self.target_angles[angle_name]
            user_angle = user_angles[angle_name]
            
            if target_angle is None or user_angle is None:
                continue
            
            tolerance = self.angle_tolerances.get(angle_name, 30.0)
            
            # For balance poses, be extra lenient with supporting leg
            if self.pose_type_detected == "balance_pose":
                if "knee" in angle_name:
                    tolerance *= 1.5
            
            difference = abs(target_angle - user_angle)
            
            if difference > tolerance:
                direction = "more" if user_angle < target_angle else "less"
                instruction = self._get_instruction(angle_name, direction, difference)
                
                if difference > tolerance * 1.5:  # Major difference
                    feedback.append(instruction)
                else:  # Minor difference
                    minor_issues.append(instruction)
        
        # Only show major issues unless pose is very close
        if len(feedback) == 0 and len(minor_issues) <= 2:
            return minor_issues
        
        return feedback
    
    def _get_instruction(self, angle_name: str, direction: str, difference: float) -> str:
        """
        Convert angle differences to human-readable instructions.
        
        Args:
            angle_name: Name of the angle to adjust
            direction: Direction of adjustment ('more' or 'less')
            difference: Degree difference from target
            
        Returns:
            Human-readable instruction string
        """
        instructions = {
            'right_arm': {
                'more': "Bend your right arm a bit more",
                'less': "Straighten your right arm slightly"
            },
            'left_arm': {
                'more': "Bend your left arm a bit more", 
                'less': "Straighten your left arm slightly"
            },
            'right_knee': {
                'more': "Bend your right knee more",
                'less': "Straighten your right leg" if self.pose_type_detected != "balance_pose" else "Adjust your right leg position"
            },
            'left_knee': {
                'more': "Bend your left knee more",
                'less': "Straighten your left leg" if self.pose_type_detected != "balance_pose" else "Adjust your left leg position"
            },
            'right_shoulder': {
                'more': "Lift your right shoulder slightly",
                'less': "Relax your right shoulder"
            },
            'left_shoulder': {
                'more': "Lift your left shoulder slightly", 
                'less': "Relax your left shoulder"
            }
        }
        
        base_instruction = instructions.get(angle_name, {}).get(direction, f"Adjust {angle_name}")
        
        # Don't show exact degree differences for minor adjustments
        if difference < 20:
            return base_instruction
        else:
            return f"{base_instruction} ({difference:.1f}° off)"