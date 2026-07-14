# Configuration for Lane Detection System
# All parameters are hardcoded here

import os

# ============================================================================
# FILE PATHS (HARDCODED)
# ============================================================================
INPUT_VIDEO_FOLDER = r"C:\data\videos"  # Folder containing all videos
CALIBRATION_IMAGE_PATH = r"C:\data\calibration.jpg"  # Calibration white sheet image
OUTPUT_FOLDER = r"C:\data\processing"  # Output folder for results

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "annotated_frames"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "annotated_videos"), exist_ok=True)

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================
FRAME_SAMPLING = 5  # Process every Nth frame (1 = all frames, 5 = every 5th frame)
SAVE_FRAME_INTERVAL = 10  # Save annotated frames every Nth processed frame
OUTPUT_ANNOTATED_VIDEO = True  # Generate annotated videos
OUTPUT_SAMPLE_FRAMES = True  # Save sample frames
CSV_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "lane_measurements.csv")

# ============================================================================
# CALIBRATION PARAMETERS
# ============================================================================
SHEET_SIZE_CM = 200  # 2m x 2m white sheet = 200cm x 200cm
AUTO_DETECT_WHITE_SHEET = True  # Try automatic detection first
MANUAL_CONFIRMATION = True  # Allow manual correction of corners
WHITE_SHEET_HSV_LOWER = (0, 0, 200)  # HSV lower threshold for white
WHITE_SHEET_HSV_UPPER = (180, 30, 255)  # HSV upper threshold for white

# ============================================================================
# LANE DETECTION PARAMETERS
# ============================================================================
MIN_LANE_WIDTH_CM = 200  # Minimum lane width (cm)
MAX_LANE_WIDTH_CM = 600  # Maximum lane width (cm)
EXPECTED_LANE_COUNTS = [2, 3, 4, 5, 6]  # Possible number of lanes
LANE_DETECTION_ROI_BOTTOM_PERCENT = 0.8  # Use bottom 80% of bird's eye view

# Hough Line Detection
HOUGH_RHO = 1  # Distance resolution (pixels)
HOUGH_THETA = 1  # Angle resolution (degrees)
HOUGH_THRESHOLD = 50  # Minimum votes to detect a line
HOUGH_MIN_LINE_LENGTH = 50  # Minimum line length (pixels)
HOUGH_MAX_LINE_GAP = 20  # Maximum gap between line points

# Edge Detection
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
CANNY_KERNEL_SIZE = 5

# Morphological Operations
MORPH_KERNEL_SIZE = 5

# ============================================================================
# DEEP LEARNING PARAMETERS
# ============================================================================
USE_DEEP_LEARNING = True  # Use DL for lane detection
DL_MODEL_TYPE = "SCNN"  # Options: "SCNN", "LaneNet", "YOLOv8-seg"
DL_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for DL predictions
DL_INPUT_SIZE = (512, 256)  # Model input size (width, height)

# ============================================================================
# TRACKING & SMOOTHING PARAMETERS
# ============================================================================
# Kalman Filter
KALMAN_PROCESS_NOISE = 0.1  # Process noise variance
KALMAN_MEASUREMENT_NOISE = 5.0  # Measurement noise variance

# Temporal Smoothing
TEMPORAL_SMOOTHING_WINDOW = 5  # Number of frames for moving average

# Polynomial Fitting
POLYNOMIAL_DEGREE = 2  # Degree of polynomial for lane fitting

# ============================================================================
# GPU/CPU SETTINGS
# ============================================================================
USE_CUDA_IF_AVAILABLE = True  # Automatically use CUDA if available
DEVICE_TYPE = None  # Will be set automatically based on availability

# ============================================================================
# VISUALIZATION & OUTPUT
# ============================================================================
ANNOTATED_FRAME_FORMAT = "PNG"  # PNG or JPG
ANNOTATED_VIDEO_CODEC = "mp4v"  # Video codec
ANNOTATED_VIDEO_FPS = 30  # Output video FPS

# Lane visualization colors (BGR format)
LANE_COLOR_PRIMARY = (0, 255, 0)  # Green for lane boundaries
LANE_CENTER_COLOR = (255, 0, 0)  # Red for lane centers
LANE_WIDTH_TEXT_COLOR = (255, 255, 255)  # White for text
LANE_COUNT_TEXT_COLOR = (0, 255, 255)  # Cyan for lane count

# ============================================================================
# CONFIDENCE SCORING
# ============================================================================
CONFIDENCE_DETECTION_WEIGHT = 0.4  # Weight for how many methods agree
CONFIDENCE_LINEARITY_WEIGHT = 0.3  # Weight for how straight lines are
CONFIDENCE_CONSISTENCY_WEIGHT = 0.3  # Weight for temporal consistency

# ============================================================================
# DEBUG & LOGGING
# ============================================================================
DEBUG_MODE = True  # Print debug information
SAVE_INTERMEDIATE_IMAGES = False  # Save bird's eye view images for debugging
