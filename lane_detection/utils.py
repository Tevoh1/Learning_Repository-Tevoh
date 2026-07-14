# Utility functions for lane detection system

import cv2
import numpy as np
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_device():
    """
    Detect and setup GPU/CPU device.
    Returns device string ('cuda' or 'cpu')
    """
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
        return 'cuda'
    else:
        logger.info("CUDA not available. Using CPU")
        return 'cpu'


def get_video_info(video_path):
    """
    Extract video information (FPS, resolution, frame count)
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Contains fps, width, height, total_frames, duration_seconds
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration_seconds': duration_seconds
    }


def read_frame_at_index(video_path, frame_index):
    """
    Read a specific frame from video by index
    
    Args:
        video_path: Path to video file
        frame_index: Frame number to read (0-indexed)
        
    Returns:
        frame: BGR image array or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None


def normalize_frame(frame):
    """
    Normalize frame for better processing
    Applies histogram equalization to handle glare and shadows
    
    Args:
        frame: BGR image
        
    Returns:
        frame: Normalized image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return normalized


def apply_perspective_transform(frame, homography_matrix, output_size):
    """
    Apply perspective transform using homography matrix (to bird's eye view)
    
    Args:
        frame: Input image
        homography_matrix: 3x3 homography matrix
        output_size: Tuple (width, height) for output bird's eye view
        
    Returns:
        warped: Bird's eye view image
    """
    warped = cv2.warpPerspective(frame, homography_matrix, output_size)
    return warped


def get_frame_sampling_indices(total_frames, sampling_interval):
    """
    Generate list of frame indices to process
    
    Args:
        total_frames: Total number of frames in video
        sampling_interval: Process every Nth frame
        
    Returns:
        list: Frame indices to process
    """
    return list(range(0, total_frames, sampling_interval))


def line_to_coefficients(line):
    """
    Convert line from Hough format to ax + by + c = 0 format
    
    Args:
        line: Hough line format [rho, theta]
        
    Returns:
        tuple: (a, b, c) coefficients
    """
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    c = -rho
    return a, b, c


def point_to_line_distance(point, line_coeffs):
    """
    Calculate perpendicular distance from point to line
    
    Args:
        point: (x, y) coordinates
        line_coeffs: (a, b, c) from ax + by + c = 0
        
    Returns:
        float: Distance
    """
    a, b, c = line_coeffs
    x, y = point
    distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return distance


def cluster_points(points, distance_threshold=10):
    """
    Cluster points that are close together
    
    Args:
        points: List of points [[x1, y1], [x2, y2], ...]
        distance_threshold: Max distance within cluster
        
    Returns:
        list: List of cluster centers
    """
    if len(points) == 0:
        return []
    
    points = np.array(points, dtype=np.float32)
    clusters = []
    used = np.zeros(len(points), dtype=bool)
    
    for i in range(len(points)):
        if used[i]:
            continue
            
        cluster = [points[i]]
        used[i] = True
        
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            distance = np.linalg.norm(points[i] - points[j])
            if distance < distance_threshold:
                cluster.append(points[j])
                used[j] = True
        
        clusters.append(np.mean(cluster, axis=0))
    
    return clusters


def angle_between_lines(theta1, theta2):
    """
    Calculate angle between two lines (in radians)
    
    Args:
        theta1, theta2: Angles in radians
        
    Returns:
        float: Angle between lines (0 to pi/2)
    """
    diff = abs(theta1 - theta2)
    return min(diff, np.pi - diff)


def filter_parallel_lines(lines, angle_tolerance=5):
    """
    Filter lines to keep only nearly parallel ones (for lane detection)
    
    Args:
        lines: List of Hough lines [[rho, theta], ...]
        angle_tolerance: Tolerance in degrees
        
    Returns:
        list: Filtered lines
    """
    if len(lines) == 0:
        return []
    
    angle_tolerance_rad = np.radians(angle_tolerance)
    filtered = [lines[0]]
    
    for line in lines[1:]:
        is_parallel = False
        for ref_line in filtered:
            angle_diff = angle_between_lines(line[1], ref_line[1])
            if angle_diff < angle_tolerance_rad:
                is_parallel = True
                break
        if is_parallel:
            filtered.append(line)
    
    return filtered


def pixels_to_cm(pixel_distance, calibration_scale):
    """
    Convert pixel distance to centimeters
    
    Args:
        pixel_distance: Distance in pixels
        calibration_scale: pixels per cm
        
    Returns:
        float: Distance in cm
    """
    return pixel_distance / calibration_scale


def cm_to_pixels(cm_distance, calibration_scale):
    """
    Convert centimeters to pixel distance
    
    Args:
        cm_distance: Distance in cm
        calibration_scale: pixels per cm
        
    Returns:
        float: Distance in pixels
    """
    return cm_distance * calibration_scale


def create_output_directories(output_folder):
    """
    Create necessary output directories
    
    Args:
        output_folder: Root output folder
    """
    subdirs = ['annotated_frames', 'annotated_videos', 'debug_images']
    for subdir in subdirs:
        path = os.path.join(output_folder, subdir)
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")


def get_timestamp():
    """
    Get current timestamp as string
    
    Returns:
        str: Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
