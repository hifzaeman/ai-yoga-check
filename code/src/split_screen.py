"""
Split Screen Pose System for AI Yoga Instructor

This module contains the main split-screen system that compares user poses
with instructor video in real-time.
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple
import mediapipe as mp

from pose_analyzer import PoseAnalyzer
from video_processor import VideoInstructorAnalyzer, VideoUtils


class SplitScreenPoseSystem:
    """
    Main system for split-screen pose comparison between user and instructor.
    """
    
    def __init__(self):
        """Initialize the split-screen pose system."""
        self.pose_analyzer = PoseAnalyzer()
        self.video_utils = VideoUtils()
        
        # System state
        self.instructor_detected = False
        self.comparison_active = False
        self.final_pose_reached = False
        self.video_ended = False
        self.last_instructor_frame = None
        self.final_pose_frame_with_landmarks = None
        
        # Video recording
        self.video_writer = None
        self.output_video_path = None
        
        # Video timing controls
        self.video_total_duration = 0
        self.video_fps = 0
        self.video_total_frames = 0
        self.current_video_time = 0
        self.final_10_percent_start_time = 0
        self.in_final_10_percent = False
        
        # Smoothing for similarity scores
        self.similarity_history = deque(maxlen=10)
        
    def analyze_instructor_video_for_final_pose(self, video_path: str) -> bool:
        """
        Pre-analyze instructor video to find final pose.
        
        Args:
            video_path: Path to instructor video
            
        Returns:
            True if analysis successful, False otherwise
        """
        print("Pre-analyzing instructor video for final pose...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Store video timing information
        self.video_total_frames = total_frames
        self.video_fps = fps
        self.video_total_duration = total_frames / fps
        self.final_10_percent_start_time = self.video_total_duration * 0.9
        
        print(f"Video duration: {self.video_total_duration:.1f}s")
        print(f"Final 10% starts at: {self.final_10_percent_start_time:.1f}s")
        
        # Look for stable pose in last 20% of video
        start_frame = int(total_frames * 0.8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        pose_candidates = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_analyzer.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                angles = self.pose_analyzer.extract_key_angles(results.pose_landmarks.landmark)
                if angles and any(angle is not None for angle in angles.values()):
                    pose_candidates.append({
                        'landmarks': results.pose_landmarks.landmark,
                        'angles': angles
                    })
        
        cap.release()
        
        if pose_candidates:
            # Use the last stable pose as target
            final_pose = pose_candidates[-1]
            self.pose_analyzer.target_pose = final_pose['landmarks']
            self.pose_analyzer.target_angles = final_pose['angles']
            
            print("Target pose extracted from instructor video!")
            print("Target angles:")
            for angle_name, angle_value in final_pose['angles'].items():
                if angle_value is not None:
                    tolerance = self.pose_analyzer.angle_tolerances.get(angle_name, 30.0)
                    print(f"  {angle_name}: {angle_value:.1f}° (tolerance: ±{tolerance:.1f}°)")
            return True
        
        print("Could not extract target pose from instructor video")
        return False
    
    def setup_video_writer(self, frame_width: int, frame_height: int, fps: float) -> None:
        """
        Set up video writer to save output.
        
        Args:
            frame_width: Width of output frames
            frame_height: Height of output frames
            fps: Frames per second for output video
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_video_path = f"pose_comparison_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_video_path, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        
        print(f"Output video will be saved as: {self.output_video_path}")
    
    def get_smoothed_similarity(self, current_similarity: float) -> float:
        """
        Smooth similarity scores to reduce jitter.
        
        Args:
            current_similarity: Current frame's similarity score
            
        Returns:
            Smoothed similarity score
        """
        self.similarity_history.append(current_similarity)
        
        if len(self.similarity_history) < 3:
            return current_similarity
        
        # Use weighted average with more weight on recent frames
        weights = np.linspace(0.5, 1.0, len(self.similarity_history))
        weighted_avg = np.average(list(self.similarity_history), weights=weights)
        
        return weighted_avg
    
    def run_split_screen_system(self, instructor_video_path: str) -> None:
        """
        Main split-screen system with strict 10% time rule.
        
        Args:
            instructor_video_path: Path to instructor video file
        """
        
        # Pre-analyze instructor video
        if not self.analyze_instructor_video_for_final_pose(instructor_video_path):
            return
        
        # Initialize video captures
        instructor_cap = cv2.VideoCapture(instructor_video_path)
        user_cap = cv2.VideoCapture(0)  # Webcam
        
        # Get video properties
        instructor_fps = instructor_cap.get(cv2.CAP_PROP_FPS)
        instructor_frame_count = int(instructor_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nStarting split-screen pose matching system")
        print(f"Instructor video: {instructor_frame_count} frames at {instructor_fps:.1f} FPS")
        print(f"User camera: Ready")
        print(f"Pose comparison will ONLY start in final 10% of video time")
        print(f"Final 10% period: {self.final_10_percent_start_time:.1f}s to {self.video_total_duration:.1f}s")
        print(f"Press 'q' to quit, 'r' to restart instructor video")
        
        # Video timing
        frame_delay = 1.0 / instructor_fps
        last_time = time.time()
        current_instructor_frame = 0
        
        while True:
            current_time = time.time()
            
            # Calculate current video time
            self.current_video_time = current_instructor_frame / instructor_fps
            
            # Check if we're in the final 10% of video time
            self.in_final_10_percent = self.current_video_time >= self.final_10_percent_start_time
            
            # Read instructor frame (with proper timing)
            if not self.video_ended:
                if current_time - last_time >= frame_delay:
                    ret_instructor, instructor_frame = instructor_cap.read()
                    if not ret_instructor:
                        # Video ended - store the last frame and stop reading
                        self.video_ended = True
                        print("Instructor video ended. Holding on final frame.")
                        if self.last_instructor_frame is not None:
                            instructor_frame = self.last_instructor_frame.copy()
                        else:
                            instructor_cap.set(cv2.CAP_PROP_POS_FRAMES, instructor_frame_count - 1)
                            ret_instructor, instructor_frame = instructor_cap.read()
                            self.last_instructor_frame = instructor_frame.copy()
                    else:
                        self.last_instructor_frame = instructor_frame.copy()
                        current_instructor_frame += 1
                        last_time = current_time
                else:
                    continue
            else:
                # Video has ended, use the last stored frame
                if self.last_instructor_frame is not None:
                    instructor_frame = self.last_instructor_frame.copy()
                else:
                    instructor_cap.set(cv2.CAP_PROP_POS_FRAMES, instructor_frame_count - 1)
                    ret_instructor, instructor_frame = instructor_cap.read()
                    self.last_instructor_frame = instructor_frame.copy()
            
            # Read user camera frame
            ret_user, user_frame = user_cap.read()
            if not ret_user:
                break
            
            # Flip user frame for mirror effect
            user_frame = cv2.flip(user_frame, 1)
            
            # Resize frames to same height
            target_height = 480
            instructor_frame = self.video_utils.resize_frame(instructor_frame, target_height)
            user_frame = self.video_utils.resize_frame(user_frame, target_height)
            
            # Only process instructor pose detection when in final 10%
            if self.in_final_10_percent and not self.final_pose_reached:
                instructor_rgb = cv2.cvtColor(instructor_frame, cv2.COLOR_BGR2RGB)
                instructor_results = self.pose_analyzer.pose.process(instructor_rgb)
                
                # Check if instructor is in final pose
                if instructor_results.pose_landmarks:
                    instructor_angles = self.pose_analyzer.extract_key_angles(instructor_results.pose_landmarks.landmark)
                    
                    if self._is_instructor_in_final_pose(instructor_angles):
                        self.final_pose_reached = True
                        self.comparison_active = True
                        print("Instructor in final pose during last 10%! Starting pose comparison...")
            
            # Only activate comparison when both conditions are met:
            # 1. We're in the final 10% of video time
            # 2. Instructor is in final pose
            self.comparison_active = self.in_final_10_percent and self.final_pose_reached
            
            # Process user pose only if comparison is active
            user_results = None
            user_angles = None
            if self.comparison_active:
                user_rgb = cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB)
                user_results = self.pose_analyzer.pose.process(user_rgb)
                
                if user_results.pose_landmarks:
                    user_angles = self.pose_analyzer.extract_key_angles(user_results.pose_landmarks.landmark)
            
            # Draw pose landmarks on both frames only when comparison is active
            instructor_frame_with_landmarks = instructor_frame.copy()
            if self.comparison_active:
                # For instructor frame, use the actual video frame with landmarks
                if not self.video_ended:
                    instructor_rgb = cv2.cvtColor(instructor_frame, cv2.COLOR_BGR2RGB)
                    instructor_results = self.pose_analyzer.pose.process(instructor_rgb)
                    if instructor_results.pose_landmarks:
                        self.pose_analyzer.mp_drawing.draw_landmarks(
                            instructor_frame_with_landmarks, instructor_results.pose_landmarks, 
                            self.pose_analyzer.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2),
                            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                        self.final_pose_frame_with_landmarks = instructor_frame_with_landmarks.copy()
                else:
                    if self.final_pose_frame_with_landmarks is not None:
                        instructor_frame_with_landmarks = self.final_pose_frame_with_landmarks.copy()
                
                if user_results and user_results.pose_landmarks:
                    self.pose_analyzer.mp_drawing.draw_landmarks(
                        user_frame, user_results.pose_landmarks, 
                        self.pose_analyzer.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2),
                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
            
            # Add labels and status
            cv2.putText(instructor_frame_with_landmarks, "INSTRUCTOR", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(user_frame, "YOU", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Show time progress and status
            if not self.in_final_10_percent:
                pass
            elif not self.final_pose_reached:
                cv2.putText(instructor_frame_with_landmarks, "FINAL 10% - Waiting for pose", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(user_frame, "Get ready...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(instructor_frame_with_landmarks, "FINAL POSE", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show pose type detected
                if self.pose_analyzer.pose_type_detected:
                    pose_type_text = f"Type: {self.pose_analyzer.pose_type_detected.replace('_', ' ').title()}"
                    cv2.putText(instructor_frame_with_landmarks, pose_type_text, (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Show pose comparison feedback on user frame
                if user_results and user_results.pose_landmarks and user_angles:
                    feedback = self.pose_analyzer.generate_feedback(user_angles)
                    raw_similarity = self.pose_analyzer.calculate_pose_similarity(user_angles)
                    smoothed_similarity = self.get_smoothed_similarity(raw_similarity)
                    
                    self._display_feedback_on_user_frame(user_frame, feedback, smoothed_similarity)
                else:
                    cv2.putText(user_frame, "No pose detected", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show video ended status if applicable
            if self.video_ended:
                cv2.putText(instructor_frame_with_landmarks, "VIDEO ENDED", 
                           (10, instructor_frame_with_landmarks.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Create split screen
            combined_frame = self.video_utils.create_split_screen(user_frame, instructor_frame_with_landmarks)
            
            # Display
            cv2.imshow('Pose Matching System - User (Left) | Instructor (Right)', combined_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Restart instructor video
                instructor_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_instructor_frame = 0
                self.current_video_time = 0
                self.comparison_active = False
                self.final_pose_reached = False
                self.video_ended = False
                self.in_final_10_percent = False
                self.last_instructor_frame = None
                self.final_pose_frame_with_landmarks = None
                self.similarity_history.clear()
                print("Restarting instructor video...")
        
        # Release everything
        instructor_cap.release()
        user_cap.release()
        cv2.destroyAllWindows()
        
    def _is_instructor_in_final_pose(self, instructor_angles: Dict[str, Optional[float]]) -> bool:
        """
        Check if instructor matches final pose.
        
        Args:
            instructor_angles: Current instructor angle measurements
            
        Returns:
            True if instructor is in final pose, False otherwise
        """
        if not self.pose_analyzer.target_angles or not instructor_angles:
            return False
        
        angle_matches = 0
        total_angles = 0
        
        for angle_name in self.pose_analyzer.target_angles:
            target_angle = self.pose_analyzer.target_angles[angle_name]
            current_angle = instructor_angles[angle_name]
            
            if target_angle is None or current_angle is None:
                continue
            
            total_angles += 1
            difference = abs(target_angle - current_angle)
            
            # Use same tolerances as user matching
            tolerance = self.pose_analyzer.angle_tolerances.get(angle_name, 30.0)
            
            if difference <= tolerance:
                angle_matches += 1
        
        # Require only 70% of angles to match (more lenient)
        match_percentage = angle_matches / max(total_angles, 1)
        return match_percentage >= 0.7
    
    def _display_feedback_on_user_frame(self, frame: np.ndarray, feedback: List[str], similarity: float) -> None:
        """
        Display feedback with better visual design and context.
        
        Args:
            frame: User frame to draw feedback on
            feedback: List of feedback instructions
            similarity: Similarity score
        """
        # Granular similarity color coding
        if similarity > 92:
            similarity_color = (0, 255, 0)      # Bright green for excellent
            status_text = "Excellent!"
        elif similarity > 87:
            similarity_color = (0, 200, 255)    # Light blue for very good
            status_text = "Very Good!"
        elif similarity > 80:
            similarity_color = (0, 165, 255)    # Orange for good
            status_text = "Good"
        elif similarity > 70:
            similarity_color = (0, 100, 255)    # Yellow for fair
            status_text = "Keep trying"
        else:
            similarity_color = (0, 0, 255)      # Red for needs work
            status_text = "Needs work"
        
        # Display similarity with status
        cv2.putText(frame, f"Match: {similarity:.1f}% ({status_text})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, similarity_color, 2)
        
        # Show pose type context
        if self.pose_analyzer.pose_type_detected:
            pose_type_display = self.pose_analyzer.pose_type_detected.replace('_', ' ').title()
            cv2.putText(frame, f"Pose: {pose_type_display}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if not feedback:
            if similarity > 85:
                cv2.putText(frame, "Perfect!", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Very close!", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(frame, "Adjust:", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show only most important feedback (max 3 items)
            for i, instruction in enumerate(feedback[:3]):
                y_pos = 150 + i * 25
                # Remove degree measurements for cleaner display
                clean_instruction = instruction.split('(')[0].strip()
                cv2.putText(frame, f"• {clean_instruction}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)