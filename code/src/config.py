"""
Configuration settings for AI Yoga Instructor
"""


class Config:
    """Configuration class containing all system settings."""
    
    # Pose Analysis Settings
    MEDIAPIPE_MODEL_COMPLEXITY = 1
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Stability Tracking
    POSE_HISTORY_MAX_LENGTH = 30
    STABILITY_THRESHOLD = 15.0
    MIN_STABLE_FRAMES = 20
    STABILITY_RATIO_THRESHOLD = 0.7
    
    # Angle Tolerances (in degrees)
    ARM_TOLERANCE = 35.0
    LEG_TOLERANCE = 45.0
    SHOULDER_TOLERANCE = 25.0
    
    # Balance Pose Adjustments
    BALANCE_POSE_TOLERANCE_MULTIPLIER = 1.5
    BALANCE_POSE_HEIGHT_THRESHOLD = 0.15
    
    # Video Processing
    TARGET_FRAME_HEIGHT = 480
    FINAL_POSE_ANALYSIS_PERCENTAGE = 0.1  # Last 10% of video
    FINAL_POSE_DETECTION_THRESHOLD = 0.8  # 80% of frames for stable pose
    
    # Similarity Scoring
    SIMILARITY_HISTORY_LENGTH = 10
    ANGLE_WEIGHTS = {
        'right_arm': 1.2,
        'left_arm': 1.2,
        'right_knee': 1.5,
        'left_knee': 1.5,
        'right_shoulder': 1.0,
        'left_shoulder': 1.0
    }
    
    # Feedback Display
    MAX_FEEDBACK_ITEMS = 3
    SIMILARITY_EXCELLENT_THRESHOLD = 92
    SIMILARITY_VERY_GOOD_THRESHOLD = 87
    SIMILARITY_GOOD_THRESHOLD = 80
    SIMILARITY_FAIR_THRESHOLD = 70
    
    # Colors (BGR format for OpenCV)
    COLOR_EXCELLENT = (0, 255, 0)      # Bright green
    COLOR_VERY_GOOD = (0, 200, 255)    # Light blue
    COLOR_GOOD = (0, 165, 255)         # Orange
    COLOR_FAIR = (0, 100, 255)         # Yellow
    COLOR_NEEDS_WORK = (0, 0, 255)     # Red
    COLOR_INSTRUCTOR = (0, 255, 0)     # Green
    COLOR_USER = (255, 0, 0)           # Blue
    COLOR_WHITE = (255, 255, 255)
    COLOR_YELLOW = (0, 255, 255)       # Yellow (for status text)
    
    # System Settings
    WEBCAM_INDEX = 0
    WINDOW_NAME = 'Pose Matching System - User (Left) | Instructor (Right)'
    
    @classmethod
    def get_angle_tolerances(cls):
        """Return the default angle tolerances dictionary."""
        return {
            'right_arm': cls.ARM_TOLERANCE,
            'left_arm': cls.ARM_TOLERANCE,
            'right_knee': cls.LEG_TOLERANCE,
            'left_knee': cls.LEG_TOLERANCE,
            'right_shoulder': cls.SHOULDER_TOLERANCE,
            'left_shoulder': cls.SHOULDER_TOLERANCE
        }
    
    @classmethod
    def get_similarity_color_and_text(cls, similarity):
        """
        Get color and status text based on similarity score.
        
        Args:
            similarity: Similarity score (0-100)
            
        Returns:
            Tuple of (color, status_text)
        """
        if similarity > cls.SIMILARITY_EXCELLENT_THRESHOLD:
            return cls.COLOR_EXCELLENT, "Excellent!"
        elif similarity > cls.SIMILARITY_VERY_GOOD_THRESHOLD:
            return cls.COLOR_VERY_GOOD, "Very Good!"
        elif similarity > cls.SIMILARITY_GOOD_THRESHOLD:
            return cls.COLOR_GOOD, "Good"
        elif similarity > cls.SIMILARITY_FAIR_THRESHOLD:
            return cls.COLOR_FAIR, "Keep trying"
        else:
            return cls.COLOR_NEEDS_WORK, "Needs work"
