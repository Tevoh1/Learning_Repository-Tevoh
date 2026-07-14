# Lane analysis and measurement module

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import cv2

logger = logging.getLogger(__name__)


class LaneAnalyzer:
    """
    Analyzes detected lanes to extract measurements
    """
    
    def __init__(self,
                 calibration_scale: float,
                 min_lane_width_cm: float = 200,
                 max_lane_width_cm: float = 600):
        """
        Initialize lane analyzer
        
        Args:
            calibration_scale: Pixels per cm
            min_lane_width_cm: Minimum expected lane width
            max_lane_width_cm: Maximum expected lane width
        """
        self.calibration_scale = calibration_scale
        self.min_lane_width_px = min_lane_width_cm * calibration_scale
        self.max_lane_width_px = max_lane_width_cm * calibration_scale
        
        logger.info(f"LaneAnalyzer initialized with lane width range: "
                   f"{min_lane_width_cm}-{max_lane_width_cm} cm")
    
    def detect_lane_count_and_widths(self, lines: List[np.ndarray],
                                     frame_height: int) -> Dict:
        """
        Detect number of lanes and measure widths from detected lines
        
        Args:
            lines: List of detected line segments [[x1, y1, x2, y2], ...]
            frame_height: Height of frame
            
        Returns:
            Dict: Contains:
                - num_lanes: Number of detected lanes
                - lane_centers: X coordinates of lane centers
                - lane_widths: Widths of lanes in pixels and cm
                - lane_boundaries: X coordinates of boundaries
                - confidence: Detection confidence
        """
        result = {
            'num_lanes': 0,
            'lane_centers': [],
            'lane_widths_px': [],
            'lane_widths_cm': [],
            'lane_boundaries': [],
            'confidence': 0.0
        }
        
        if len(lines) == 0:
            return result
        
        # Filter and cluster lines by vertical position
        vertical_positions = self._get_line_positions(lines, frame_height)
        
        if len(vertical_positions) == 0:
            return result
        
        # Sort by X position
        sorted_positions = sorted(vertical_positions, key=lambda x: x[1])
        
        # Cluster to find lane boundaries
        boundaries = self._cluster_positions(sorted_positions)
        
        # Extract lanes and widths
        result['lane_boundaries'] = boundaries
        
        if len(boundaries) >= 2:
            result['lane_centers'] = self._calculate_lane_centers(boundaries)
            result['lane_widths_px'], result['lane_widths_cm'] = \
                self._calculate_lane_widths(boundaries)
            result['num_lanes'] = len(result['lane_widths_px'])
            result['confidence'] = self._calculate_confidence(result['lane_widths_px'])
        
        return result
    
    def _get_line_positions(self, lines: List[np.ndarray],
                           frame_height: int) -> List[Tuple]:
        """
        Extract vertical line positions from detected lines
        
        Args:
            lines: Detected line segments
            frame_height: Frame height for y-interpolation
            
        Returns:
            List: [(x_position, avg_x), ...] for each line
        """
        positions = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Get average x position
            avg_x = (x1 + x2) / 2.0
            
            # Get x position at middle of frame
            if y1 != y2:
                slope = (x2 - x1) / (y2 - y1)
                x_middle = x1 + slope * (frame_height / 2 - y1)
            else:
                x_middle = avg_x
            
            positions.append((x1, x2, x_middle))
        
        return positions
    
    def _cluster_positions(self, positions: List[Tuple],
                          distance_threshold: float = 30) -> List[float]:
        """
        Cluster nearby positions (lane boundaries tend to cluster)
        
        Args:
            positions: Line positions [(x1, x2, x_middle), ...]
            distance_threshold: Maximum distance within cluster
            
        Returns:
            List: Clustered positions (lane boundaries)
        """
        if len(positions) == 0:
            return []
        
        # Use x_middle (third element) for clustering
        x_values = [p[2] for p in positions]
        x_values_sorted = sorted(x_values)
        
        clusters = []
        current_cluster = [x_values_sorted[0]]
        
        for x in x_values_sorted[1:]:
            if x - current_cluster[-1] < distance_threshold:
                current_cluster.append(x)
            else:
                # Save cluster
                clusters.append(np.mean(current_cluster))
                current_cluster = [x]
        
        # Save last cluster
        if current_cluster:
            clusters.append(np.mean(current_cluster))
        
        return sorted(clusters)
    
    def _calculate_lane_centers(self, boundaries: List[float]) -> List[float]:
        """
        Calculate lane center positions from boundaries
        
        Args:
            boundaries: Sorted lane boundary positions
            
        Returns:
            List: Lane center positions
        """
        if len(boundaries) < 2:
            return []
        
        centers = []
        for i in range(len(boundaries) - 1):
            center = (boundaries[i] + boundaries[i + 1]) / 2.0
            centers.append(center)
        
        return centers
    
    def _calculate_lane_widths(self, boundaries: List[float]) -> Tuple[List[float], List[float]]:
        """
        Calculate lane widths from boundaries
        
        Args:
            boundaries: Sorted lane boundary positions
            
        Returns:
            Tuple: (widths_px, widths_cm)
        """
        widths_px = []
        widths_cm = []
        
        for i in range(len(boundaries) - 1):
            width_px = boundaries[i + 1] - boundaries[i]
            
            # Validate width
            if self.min_lane_width_px <= width_px <= self.max_lane_width_px:
                widths_px.append(width_px)
                widths_cm.append(width_px / self.calibration_scale)
        
        return widths_px, widths_cm
    
    def _calculate_confidence(self, widths_px: List[float]) -> float:
        """
        Calculate confidence based on measurements
        
        Args:
            widths_px: Detected lane widths in pixels
            
        Returns:
            float: Confidence score (0.0-1.0)
        """
        if len(widths_px) == 0:
            return 0.0
        
        # Confidence based on consistency of widths
        if len(widths_px) == 1:
            confidence = 0.6
        else:
            # Calculate coefficient of variation
            mean_width = np.mean(widths_px)
            std_width = np.std(widths_px)
            cv = std_width / mean_width if mean_width > 0 else 1.0
            
            # Confidence decreases with variation
            confidence = max(0.0, 1.0 - cv)
        
        return confidence
    
    def filter_detections(self, detections: List[Dict],
                         min_frame_consistency: int = 2) -> List[Dict]:
        """
        Filter detections based on consistency across frames
        
        Args:
            detections: List of frame detections
            min_frame_consistency: Minimum frames to consider valid
            
        Returns:
            List: Filtered detections
        """
        if len(detections) < min_frame_consistency:
            return detections
        
        # For now, return all detections
        # Could implement temporal filtering here
        return detections
