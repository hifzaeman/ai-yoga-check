"""
AI Yoga Instructor - Main Application

A split-screen pose matching system that compares user poses with instructor videos
in real-time using computer vision and pose estimation.
"""

import sys
import time
from pathlib import Path

from src.split_screen import SplitScreenPoseSystem
from src.ideo_processor import VideoInstructorAnalyzer, VideoUtils
from src.config import Config


def print_welcome():
    """Print welcome message and feature overview."""
    print("=" * 60)
    print("        AI YOGA INSTRUCTOR - POSE MATCHING SYSTEM")
    print("=" * 60)
    print()
    print("FEATURES:")
    print("• Split-screen view: Instructor video (RIGHT) | Your camera (LEFT)")
    print("• Analyzes FINAL POSE from last 10% of instructor video")
    print("• Comparison ONLY starts in final 10% of video time")
    print("• Adaptive tolerances based on pose type (balance vs standard)")
    print("• Real-time feedback with weighted angle scoring")
    print("• Smoothed similarity scoring to reduce jitter")
    print("• Video holds on final frame when ended (no looping)")
    print()
    print("TOLERANCE SETTINGS:")
    print(f"   • Arms: ±{Config.ARM_TOLERANCE}° (flexible for different positions)")
    print(f"   • Legs: ±{Config.LEG_TOLERANCE}° (extra lenient for balance poses)")
    print(f"   • Shoulders: ±{Config.SHOULDER_TOLERANCE}° (moderate precision)")
    print("   • Balance poses get 50% extra tolerance automatically")
    print()


def get_video_path() -> str:
    """
    Get and validate instructor video path from user.
    
    Returns:
        Valid video file path
    """
    while True:
        video_path = input("Enter instructor video file path: ").strip()
        
        if not video_path:
            print("Please enter a valid path.")
            continue
            
        # Remove quotes if present
        video_path = video_path.strip('"\'')
        
        # Check if file exists
        if not Path(video_path).exists():
            print(f"File not found: {video_path}")
            continue
        
        # Validate video file
        is_valid, message = VideoUtils.validate_video_file(video_path)
        if not is_valid:
            print(f"Invalid video file: {message}")
            continue
            
        # Get video info
        video_info = VideoUtils.get_video_info(video_path)
        if video_info:
            print(f"\nVideo Information:")
            print(f"  Duration: {video_info['duration']:.1f} seconds")
            print(f"  Resolution: {video_info['width']}x{video_info['height']}")
            print(f"  FPS: {video_info['fps']:.1f}")
            print(f"  Total frames: {video_info['frame_count']}")
            
            if video_info['duration'] < 10:
                print("Warning: Video is very short. Consider using a longer video for better results.")
            
        return video_path


def run_pose_analysis(video_path: str) -> bool:
    """
    Run the pose analysis on the instructor video.
    
    Args:
        video_path: Path to instructor video
        
    Returns:
        True if analysis successful, False otherwise
    """
    print("\n" + "=" * 60)
    print("ANALYZING INSTRUCTOR VIDEO...")
    print("=" * 60)
    
    analyzer = VideoInstructorAnalyzer()
    
    try:
        success = analyzer.extract_final_pose_from_video(video_path)
        if not success:
            print("Failed to analyze instructor video.")
            print("Make sure the video contains clear pose demonstrations.")
            return False
            
        return True, analyzer
        
    except Exception as e:
        print(f"Error during video analysis: {str(e)}")
        return False


def run_split_screen_system(video_path: str, analyzer: VideoInstructorAnalyzer):
    """
    Run the main split-screen pose matching system.
    
    Args:
        video_path: Path to instructor video
        analyzer: Configured video analyzer with target pose
    """
    print("\n" + "=" * 60)
    print("STARTING SPLIT-SCREEN POSE MATCHING")
    print("=" * 60)
    print("Instructor video (RIGHT) | Your camera (LEFT)")
    print("Comparison starts ONLY in final 10% of video time")
    print("AI will detect pose type and adjust tolerances automatically")
    print("Press 'q' to quit, 'r' to restart video")
    print("=" * 60)
    
    # Initialize system
    system = SplitScreenPoseSystem()
    
    # Transfer target pose from analyzer to main system
    system.pose_analyzer.target_pose = analyzer.pose_analyzer.target_pose
    system.pose_analyzer.target_angles = analyzer.pose_analyzer.target_angles
    system.pose_analyzer.angle_tolerances = analyzer.pose_analyzer.angle_tolerances
    
    # Give user time to prepare
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    try:
        system.run_split_screen_system(video_path)
        
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
    except Exception as e:
        print(f"\nError during pose matching: {str(e)}")
    finally:
        print("\nPose matching system stopped.")


def main():
    """Main application entry point."""
    try:
        # Print welcome message
        print_welcome()
        
        # Get video path from user
        video_path = get_video_path()
        
        # Analyze instructor video
        result = run_pose_analysis(video_path)
        if isinstance(result, tuple) and result[0]:
            success, analyzer = result
        else:
            print("Exiting due to video analysis failure.")
            return 1
        
        # Run split-screen system
        run_split_screen_system(video_path, analyzer)
        
        print("\nThank you for using AI Yoga Instructor!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        return 1


def debug_tolerances():
    """Debug function to display current angle tolerances."""
    from src.pose_analyzer import PoseAnalyzer
    
    print("=== CURRENT ANGLE TOLERANCES ===")
    analyzer = PoseAnalyzer()
    
    for angle_name, tolerance in analyzer.angle_tolerances.items():
        print(f"{angle_name:15}: ±{tolerance:5.1f}°")
    
    print("\n=== POSE TYPE ADJUSTMENTS ===")
    print("Standard pose: Default tolerances")
    print("Balance pose:  Knee tolerances +50% (±67.5°)")
    print("               Stability threshold +50%")


if __name__ == "__main__":
    # Uncomment the line below to see current tolerance settings
    # debug_tolerances()
    
    sys.exit(main())