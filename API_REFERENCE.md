# API Reference

Detailed documentation of all classes and methods in the lane detection system.

## Lane Detection Module (`lane_detection.py`)

### TraditionalLaneDetector

```python
class TraditionalLaneDetector:
    def __init__(self,
                 canny_low: int = 50,
                 canny_high: int = 150,
                 hough_rho: float = 1,
                 hough_theta: float = 1,
                 hough_threshold: int = 50,
                 min_line_length: int = 50,
                 max_line_gap: int = 20,
                 morph_kernel_size: int = 5)
    
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """Detect lanes using Canny + Hough
        
        Args:
            frame: Input BGR image
            
        Returns:
            - detected_lines: List of [x1, y1, x2, y2] line segments
            - edge_map: Edge detection result
        """
```

### DeepLearningLaneDetector

```python
class DeepLearningLaneDetector:
    def __init__(self, model_type: str = 'SCNN', device: str = 'cpu'):
        """Initialize DL detector
        
        Args:
            model_type: 'SCNN', 'LaneNet', etc.
            device: 'cuda' or 'cpu'
        """
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray], Optional[float]]:
        """Detect lanes using deep learning
        
        Returns:
            - lines: Detected lane lines or None
            - segmentation: Segmentation mask or None
            - confidence: Confidence score or None
        """
```

### HybridLaneDetector

```python
class HybridLaneDetector:
    def __init__(self,
                 use_traditional: bool = True,
                 use_deep_learning: bool = True,
                 device: str = 'cpu',
                 **cv_params)
    
    def detect(self, frame: np.ndarray,
               roi_mask: Optional[np.ndarray] = None) -> Dict:
        """Detect lanes using hybrid approach
        
        Args:
            frame: Input bird's eye view image
            roi_mask: Optional ROI mask
            
        Returns:
            Dict containing:
            - cv_lines: Traditional CV detections
            - dl_lines: DL detections
            - all_lines: Combined
            - confidence: Overall confidence
            - edge_map: Edge detection result
        """
```

## Calibration Module (`calibration.py`)

### CalibrationManager

```python
class CalibrationManager:
    def __init__(self, calibration_image_path: str, sheet_size_cm: float = 200):
        """Initialize calibration manager
        
        Args:
            calibration_image_path: Path to calibration image
            sheet_size_cm: Size of white sheet in cm
        """
    
    def auto_detect_white_sheet(self,
                               hsv_lower: Tuple[int, int, int] = (0, 0, 200),
                               hsv_upper: Tuple[int, int, int] = (180, 30, 255)
                               ) -> Optional[np.ndarray]:
        """Automatically detect white sheet corners
        
        Returns:
            4 corner points or None if failed
        """
    
    def manual_select_corners(self) -> np.ndarray:
        """Manually select corners by clicking
        
        Returns:
            4 corner points
        """
    
    def calibrate(self,
                 use_auto_detect: bool = True,
                 manual_confirm: bool = False) -> Tuple[np.ndarray, float]:
        """Run calibration pipeline
        
        Returns:
            - homography_matrix: 3x3 transformation matrix
            - calibration_scale: pixels per cm
        """
    
    def get_bird_eye_size(self) -> Tuple[int, int]:
        """Get output bird's eye view size
        
        Returns:
            (width, height) in pixels
        """
```

## Perspective Transform Module (`perspective_transform.py`)

### PerspectiveTransformer

```python
class PerspectiveTransformer:
    def __init__(self,
                 homography_matrix: np.ndarray,
                 output_size: Tuple[int, int],
                 calibration_scale: float)
    
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """Transform frame to bird's eye view
        
        Returns:
            Transformed frame
        """
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform 2D points
        
        Args:
            points: Shape (N, 2) or (N, 1, 2)
            
        Returns:
            Transformed points with same shape
        """
    
    def pixel_to_cm(self, pixel_distance: float) -> float:
        """Convert pixels to centimeters"""
    
    def cm_to_pixel(self, cm_distance: float) -> float:
        """Convert centimeters to pixels"""
    
    def get_roi_mask(self, roi_bottom_percent: float = 0.8) -> np.ndarray:
        """Get region of interest mask
        
        Args:
            roi_bottom_percent: Fraction of image to use (0.0-1.0)
            
        Returns:
            Binary mask
        """
```

## Lane Tracking Module (`lane_tracking.py`)

### KalmanFilter1D

```python
class KalmanFilter1D:
    def __init__(self,
                 process_noise: float = 0.1,
                 measurement_noise: float = 5.0)
    
    def initialize(self, measurement: float):
        """Initialize with first measurement"""
    
    def predict(self):
        """Predict next state"""
    
    def update(self, measurement: float):
        """Update with new measurement"""
    
    def get_state(self) -> float:
        """Get current estimated state"""
```

### LaneTracker

```python
class LaneTracker:
    def __init__(self,
                 kalman_process_noise: float = 0.1,
                 kalman_measurement_noise: float = 5.0,
                 temporal_smoothing_window: int = 5)
    
    def initialize_lane(self, lane_id: int, initial_position: float):
        """Initialize tracking for new lane"""
    
    def update_lane(self, lane_id: int,
                    measurement: float) -> Dict[str, float]:
        """Update lane tracking
        
        Returns:
            Dict with keys: 'raw', 'kalman', 'temporal_avg'
        """
    
    def reset(self):
        """Reset all tracking data"""
```

### LanePolynomialFitter

```python
class LanePolynomialFitter:
    def __init__(self, degree: int = 2, window_size: int = 5)
    
    def add_measurement(self, lane_id: int,
                       x_coords: np.ndarray,
                       y_coords: np.ndarray):
        """Add lane boundary measurements"""
    
    def fit_lane(self, lane_id: int) -> Optional[np.ndarray]:
        """Fit polynomial to lane
        
        Returns:
            Polynomial coefficients or None
        """
    
    def evaluate_polynomial(self, coeffs: np.ndarray,
                           y_values: np.ndarray) -> np.ndarray:
        """Evaluate polynomial at y positions"""
```

## Lane Analyzer Module (`lane_analyzer.py`)

### LaneAnalyzer

```python
class LaneAnalyzer:
    def __init__(self,
                 calibration_scale: float,
                 min_lane_width_cm: float = 200,
                 max_lane_width_cm: float = 600)
    
    def detect_lane_count_and_widths(self,
                                    lines: List[np.ndarray],
                                    frame_height: int) -> Dict:
        """Detect lanes and measure widths
        
        Returns:
            Dict containing:
            - num_lanes: Number of lanes
            - lane_centers: X positions of centers
            - lane_widths_px: Widths in pixels
            - lane_widths_cm: Widths in centimeters
            - lane_boundaries: X positions of boundaries
            - confidence: Detection confidence
        """
```

## Video Processor Module (`video_processor.py`)

### VideoProcessor

```python
class VideoProcessor:
    def __init__(self,
                 calibration_manager: CalibrationManager,
                 config: Dict)
    
    def process_video(self,
                     video_path: str,
                     frame_sampling: int = 1,
                     save_frames_callback=None,
                     save_video_callback=None) -> List[Dict]:
        """Process video and extract measurements
        
        Args:
            video_path: Path to video file
            frame_sampling: Process every Nth frame
            save_frames_callback: Callback(original, bev, frame_num)
            save_video_callback: Callback(frames, video_info)
            
        Returns:
            List of frame measurements
        """
    
    def set_current_video_path(self, video_path: str):
        """Set current video path for reference"""
```

## Output Handler Module (`output_handler.py`)

### OutputHandler

```python
class OutputHandler:
    def __init__(self, output_folder: str, config: Dict)
    
    def add_measurements(self, measurements: List[Dict],
                        video_name: str):
        """Add frame measurements to output buffer"""
    
    def save_csv(self):
        """Save all measurements to CSV"""
    
    def save_annotated_frames(self,
                             original_frame: np.ndarray,
                             bev_frame: np.ndarray,
                             frame_number: int,
                             video_name: str):
        """Save annotated frame (side-by-side)"""
    
    def save_annotated_video(self,
                            frames: List[Tuple[np.ndarray, np.ndarray]],
                            video_info: Dict,
                            video_name: str):
        """Save annotated video with side-by-side views"""
    
    def generate_statistics_report(self,
                                  measurements: List[Dict]) -> Dict:
        """Generate statistics from measurements
        
        Returns:
            Dict with statistics keys
        """
```

## Utils Module (`utils.py`)

```python
def setup_device() -> str:
    """Detect and setup GPU/CPU device
    
    Returns:
        'cuda' or 'cpu'
    """

def get_video_info(video_path: str) -> Dict:
    """Extract video information
    
    Returns:
        Dict with: fps, width, height, total_frames, duration_seconds
    """

def read_frame_at_index(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Read specific frame from video"""

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame for better processing (CLAHE)"""

def get_frame_sampling_indices(total_frames: int,
                              sampling_interval: int) -> List[int]:
    """Generate list of frame indices to process"""

def pixels_to_cm(pixel_distance: float,
                calibration_scale: float) -> float:
    """Convert pixel distance to cm"""

def cm_to_pixels(cm_distance: float,
                calibration_scale: float) -> float:
    """Convert cm to pixel distance"""
```

---

For more information, see the docstrings in individual modules.
