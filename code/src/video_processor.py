"""
Video Processing Module for AI Yoga Instructor

This module handles video analysis, final pose extraction, and video processing
utilities for the AI yoga instructor system.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pose_analyzer import PoseAnalyzer


class VideoInstructorAnalyzer:
    """
    Analyzes instructor videos to extract final poses and target positions.
    """
    
    def __init__(self):
        """Initialize the video analyzer with a pose analyzer instance."""
        self.pose_analyzer = PoseAnalyzer()
        
    def extract_final_pose_from_video(self, video_path: str) -> bool:
        """
        Extract final pose from the last 10% time duration of instructor video.
        
        Args:
            video_path: Path to the instructor video file
            
        Returns:
            True if final pose extracted successfully, False otherwise
        """
        print(f"Analyzing FINAL POSE from last 10% TIME of instructor video...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Could not open video file. Please check the path.")
            return False
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            print("Invalid FPS value.")
            cap.release()
            return False
        
        # Calculate time-based analysis
        total_duration = total_frames / fps
        last_10_percent_duration = total_duration * 0.1
        start_time = total_duration - last_10_percent_duration
        start_frame = int(start_time * fps)
        
        print(f"Video: {total_duration:.1f}s total duration")
        print(f"Analyzing last {last_10_percent_duration:.1f}s (from {start_time:.1f}s to {total_duration:.1f}s)")
        print(f"Frame range: {start_frame} to {total_frames} ({total_frames - start_frame} frames)")
        
        # Jump to the last 10% TIME point
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        final_poses = []
        frame_number = start_frame
        current_time = start_time
        
        while cap.isOpened() and current_time < total_duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_analyzer.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                angles = self.pose_analyzer.extract_key_angles(results.pose_landmarks.landmark)
                if angles and any(angle is not None for angle in angles.values()):
                    final_poses.append({
                        'frame': frame_number,
                        'timestamp': current_time,
                        'landmarks': results.pose_landmarks.landmark,
                        'angles': angles
                    })
                    
                    # Show progress every 10 frames
                    if len(final_poses) % 10 == 0:
                        progress = ((current_time - start_time) / last_10_percent_duration) * 100
                        print(f"  Progress: {progress:.1f}% - Time: {current_time:.1f}s")
            
            frame_number += 1
            current_time = frame_number / fps
        
        cap.release()
        
        print(f"Analyzed final {len(final_poses)} poses from last 10% of video time")
        
        if len(final_poses) < 5:
            print("Not enough valid poses found in final 10% TIME of video")
            return False
        
        # Find the most stable pose in the final 10% TIME
        final_pose = self._find_most_stable_final_pose(final_poses)
        
        if final_pose:
            self.pose_analyzer.target_pose = final_pose['landmarks']
            self.pose_analyzer.target_angles = final_pose['angles']
            
            print(f"FINAL POSE detected at {final_pose['timestamp']:.1f}s")
            print("Target angles for FINAL POSE:")
            for angle_name, angle_value in final_pose['angles'].items():
                if angle_value is not None:
                    tolerance = self.pose_analyzer.angle_tolerances.get(angle_name, 30.0)
                    print(f"   {angle_name}: {angle_value:.1f}° (±{tolerance:.1f}°)")
            return True
        
        print("Could not identify FINAL POSE from last 10% TIME of video")
        return False
    
    def _find_most_stable_final_pose(self, final_poses: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the most stable pose in the final 10% of video.
        
        Args:
            final_poses: List of pose candidates
            
        Returns:
            The most stable pose or None if not found
        """
        if len(final_poses) < 10:
            return final_poses[-1] if final_poses else None
        
        window_size = min(15, len(final_poses) // 2)
        best_stability = float('inf')
        best_pose = None
        
        print("Searching for most stable FINAL POSE in last 10%...")
        
        for i in range(len(final_poses) - window_size):
            window = final_poses[i:i + window_size]
            stability_score = self._calculate_stability_for_final_pose(window)
            
            if stability_score < best_stability:
                best_stability = stability_score
                best_pose = window[len(window)//2]  # Take middle pose of stable sequence
        
        if best_pose:
            print(f"Most stable FINAL POSE found with stability score: {best_stability:.2f}")
        
        return best_pose
    
    def _calculate_stability_for_final_pose(self, pose_window: List[Dict[str, Any]]) -> float:
        """
        Calculate how stable the final pose sequence is.
        
        Args:
            pose_window: Window of poses to analyze for stability
            
        Returns:
            Stability score (lower is better)
        """
        if len(pose_window) < 2:
            return float('inf')
        
        total_variation = 0
        angle_count = 0
        
        for angle_name in pose_window[0]['angles']:
            angle_values = [pose['angles'][angle_name] for pose in pose_window 
                           if pose['angles'][angle_name] is not None]
            
            if len(angle_values) > 1:
                total_variation += np.std(angle_values)
                angle_count += 1
        
        return total_variation / max(angle_count, 1)


class VideoUtils:
    """
    Utility functions for video processing and frame manipulation.
    """
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_height: int) -> Optional[np.ndarray]:
        """
        Resize frame maintaining aspect ratio.
        
        Args:
            frame: Input frame
            target_height: Desired height
            
        Returns:
            Resized frame or None if input is invalid
        """
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        target_width = int(target_height * aspect_ratio)
        return cv2.resize(frame, (target_width, target_height))
    
    @staticmethod
    def create_split_screen(left_frame: np.ndarray, right_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Create side-by-side split screen.
        
        Args:
            left_frame: Frame for left side
            right_frame: Frame for right side
            
        Returns:
            Combined split-screen frame or None if inputs invalid
        """
        if left_frame is None or right_frame is None:
            return None
            
        height = min(left_frame.shape[0], right_frame.shape[0])
        left_frame = cv2.resize(left_frame, (left_frame.shape[1], height))
        right_frame = cv2.resize(right_frame, (right_frame.shape[1], height))
        
        # Add separator line
        separator = np.zeros((height, 5, 3), dtype=np.uint8)
        separator[:, :] = [255, 255, 255]  # White separator
        
        # Combine frames
        combined = np.hstack([left_frame, separator, right_frame])
        return combined
    
    @staticmethod
    def validate_video_file(video_path: str) -> Tuple[bool, str]:
        """
        Validate if video file can be opened and read.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Could not open video file. Please check the path."
            
            # Try to read first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False, "Could not read from video file. File may be corrupted."
            
            # Check basic properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            if fps <= 0:
                return False, "Invalid video FPS value."
            
            if frame_count <= 0:
                return False, "Invalid video frame count."
            
            return True, "Video file is valid"
            
        except Exception as e:
            return False, f"Error validating video: {str(e)}"
    
    @staticmethod
    def get_video_info(video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if error
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
            
            # Calculate duration
            info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
            
            cap.release()
            return info
            
        except Exception:
            return None