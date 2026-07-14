# Lane tracking and smoothing module

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class KalmanFilter1D:
    """
    Simple 1D Kalman filter for tracking scalar values (lane positions)
    """
    
    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 5.0):
        """
        Initialize 1D Kalman filter
        
        Args:
            process_noise: Process noise variance (Q)
            measurement_noise: Measurement noise variance (R)
        """
        self.Q = process_noise  # Process noise
        self.R = measurement_noise  # Measurement noise
        
        # State
        self.x = None  # Estimated position
        self.P = 1.0  # Estimate error
    
    def initialize(self, measurement: float):
        """
        Initialize filter with first measurement
        
        Args:
            measurement: Initial measurement
        """
        self.x = measurement
        self.P = 1.0
    
    def predict(self):
        """
        Predict next state (assume constant velocity model)
        """
        # x remains the same (no velocity)
        self.P = self.P + self.Q
    
    def update(self, measurement: float):
        """
        Update with measurement
        
        Args:
            measurement: New measurement
        """
        # Kalman gain
        K = self.P / (self.P + self.R)
        
        # Update estimate
        self.x = self.x + K * (measurement - self.x)
        
        # Update error estimate
        self.P = (1 - K) * self.P
    
    def get_state(self) -> float:
        """
        Get current estimated state
        
        Returns:
            float: Estimated value
        """
        return self.x


class LaneTracker:
    """
    Tracks detected lanes across frames
    """
    
    def __init__(self,
                 kalman_process_noise: float = 0.1,
                 kalman_measurement_noise: float = 5.0,
                 temporal_smoothing_window: int = 5):
        """
        Initialize lane tracker
        
        Args:
            kalman_process_noise: Kalman filter Q parameter
            kalman_measurement_noise: Kalman filter R parameter
            temporal_smoothing_window: Window size for temporal smoothing
        """
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.temporal_smoothing_window = temporal_smoothing_window
        
        # Tracking data per lane
        self.kalman_filters: Dict[int, KalmanFilter1D] = {}
        self.temporal_history: Dict[int, deque] = {}  # Store past measurements
        
        logger.info(f"LaneTracker initialized with smoothing window {temporal_smoothing_window}")
    
    def initialize_lane(self, lane_id: int, initial_position: float):
        """
        Initialize tracking for a new lane
        
        Args:
            lane_id: Unique lane identifier
            initial_position: Initial lane position
        """
        kf = KalmanFilter1D(self.kalman_process_noise, self.kalman_measurement_noise)
        kf.initialize(initial_position)
        self.kalman_filters[lane_id] = kf
        self.temporal_history[lane_id] = deque(maxlen=self.temporal_smoothing_window)
        self.temporal_history[lane_id].append(initial_position)
    
    def update_lane(self, lane_id: int, measurement: float) -> Dict[str, float]:
        """
        Update tracking for a lane with new measurement
        
        Args:
            lane_id: Lane identifier
            measurement: New position measurement
            
        Returns:
            Dict: Tracking results including raw, kalman, and smoothed values
        """
        if lane_id not in self.kalman_filters:
            self.initialize_lane(lane_id, measurement)
        
        # Kalman filter update
        kf = self.kalman_filters[lane_id]
        kf.predict()
        kf.update(measurement)
        kalman_result = kf.get_state()
        
        # Temporal smoothing (moving average)
        self.temporal_history[lane_id].append(measurement)
        temporal_result = np.mean(list(self.temporal_history[lane_id]))
        
        return {
            'raw': measurement,
            'kalman': kalman_result,
            'temporal_avg': temporal_result
        }
    
    def reset(self):
        """
        Reset all tracking data (for new video or major scene change)
        """
        self.kalman_filters.clear()
        self.temporal_history.clear()
        logger.info("Lane tracker reset")


class LanePolynomialFitter:
    """
    Fits polynomial curves to lane boundaries
    """
    
    def __init__(self, degree: int = 2, window_size: int = 5):
        """
        Initialize polynomial fitter
        
        Args:
            degree: Polynomial degree
            window_size: Number of frames for polynomial fitting
        """
        self.degree = degree
        self.window_size = window_size
        self.history: Dict[int, deque] = {}  # Store measurements per lane
        
        logger.info(f"LanePolynomialFitter initialized with degree {degree}")
    
    def add_measurement(self, lane_id: int, x_coords: np.ndarray, y_coords: np.ndarray):
        """
        Add lane boundary measurements
        
        Args:
            lane_id: Lane identifier
            x_coords: X coordinates of lane boundary
            y_coords: Y coordinates of lane boundary
        """
        if lane_id not in self.history:
            self.history[lane_id] = deque(maxlen=self.window_size)
        
        self.history[lane_id].append({
            'x': x_coords,
            'y': y_coords
        })
    
    def fit_lane(self, lane_id: int) -> Optional[np.ndarray]:
        """
        Fit polynomial to lane boundary
        
        Args:
            lane_id: Lane identifier
            
        Returns:
            np.ndarray: Polynomial coefficients or None if insufficient data
        """
        if lane_id not in self.history or len(self.history[lane_id]) == 0:
            return None
        
        # Collect all points from history
        all_x = []
        all_y = []
        for measurement in self.history[lane_id]:
            all_x.extend(measurement['x'])
            all_y.extend(measurement['y'])
        
        if len(all_x) < self.degree + 1:
            return None
        
        # Fit polynomial
        coeffs = np.polyfit(all_y, all_x, self.degree)
        return coeffs
    
    def evaluate_polynomial(self, coeffs: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at y positions
        
        Args:
            coeffs: Polynomial coefficients
            y_values: Y coordinates to evaluate at
            
        Returns:
            np.ndarray: X coordinates
        """
        return np.polyval(coeffs, y_values)
    
    def reset(self):
        """
        Reset all data
        """
        self.history.clear()
        logger.info("Polynomial fitter reset")


class LaneAssociator:
    """
    Associates detected lines across frames to track lanes
    """
    
    def __init__(self, distance_threshold: float = 50):
        """
        Initialize lane associator
        
        Args:
            distance_threshold: Maximum distance for associating lines
        """
        self.distance_threshold = distance_threshold
        self.lane_assignments: Dict[int, List] = {}  # lane_id -> list of lines
        self.next_lane_id = 0
    
    def associate_lines(self, current_lines: List[np.ndarray],
                       previous_positions: Optional[Dict[int, float]] = None) -> Dict[int, np.ndarray]:
        """
        Associate current detected lines with tracked lanes
        
        Args:
            current_lines: Current frame's detected lines
            previous_positions: Previous frame's lane positions
            
        Returns:
            Dict: lane_id -> line mapping
        """
        associations = {}
        
        if previous_positions is None or len(previous_positions) == 0:
            # New lanes
            for line in current_lines:
                lane_id = self.next_lane_id
                self.lane_assignments[lane_id] = [line]
                associations[lane_id] = line
                self.next_lane_id += 1
        else:
            # Try to match with previous lanes
            used_lines = set()
            
            for lane_id, prev_pos in previous_positions.items():
                best_match = None
                best_distance = self.distance_threshold
                best_idx = None
                
                for idx, line in enumerate(current_lines):
                    if idx in used_lines:
                        continue
                    
                    # Calculate distance from previous position
                    line_center = np.mean(line[:2])  # Average of start and end x
                    distance = abs(line_center - prev_pos)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = line
                        best_idx = idx
                
                if best_match is not None:
                    used_lines.add(best_idx)
                    associations[lane_id] = best_match
                    self.lane_assignments[lane_id].append(best_match)
            
            # New lanes from unmatched lines
            for idx, line in enumerate(current_lines):
                if idx not in used_lines:
                    lane_id = self.next_lane_id
                    self.lane_assignments[lane_id] = [line]
                    associations[lane_id] = line
                    self.next_lane_id += 1
        
        return associations
    
    def reset(self):
        """
        Reset association data
        """
        self.lane_assignments.clear()
        self.next_lane_id = 0
