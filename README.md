**AI Yoga Instructor - Pose Matching System**
A real-time pose comparison system that uses computer vision to analyze yoga poses and provide feedback by comparing user movements with instructor demonstration videos.
Features

Split-Screen Interface: Real-time comparison between instructor video (right) and user camera (left)
Intelligent Pose Analysis: Analyzes final poses from the last 10% of instructor videos
Adaptive Tolerances: Different tolerance levels for arms, legs, and shoulders with automatic adjustments for balance poses
Real-Time Feedback: Provides specific instructions for pose corrections
Smart Pose Detection: Automatically detects pose types (standard vs balance poses)
Smoothed Scoring: Reduces jitter in similarity calculations for stable feedback

**Installation**

**Clone the repository:**

bashgit clone https://github.com/yourusername/ai-yoga-instructor.git
cd ai-yoga-instructor

**Install required dependencies:**

bashpip install -r requirements.txt
Usage

Run the main application:

bashpython main.py

When prompted, enter the path to your instructor video file
The system will analyze the video and extract the final pose
Position yourself in front of your camera and follow the on-screen instructions

**Controls**

'q': Quit the application
'r': Restart the instructor video

**System Requirements**

Python 3.7 or higher
Webcam for user pose detection
Video file containing yoga pose demonstration


'r': Restart the instructor video

System Requirements

Python 3.7 or higher
Webcam for user pose detection
Video file containing yoga pose demonstration
