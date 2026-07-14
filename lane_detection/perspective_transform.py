# Perspective transformation utilities

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class PerspectiveTransformer:
    """
    Handles perspective transformation from camera view to bird's eye view
    """
    
    def __init__(self, homography_matrix: np.ndarray, 
                 output_size: Tuple[int, int],
                 calibration_scale: float):
        """
        Initialize perspective transformer
        
        Args:
            homography_matrix: 3x3 homography matrix from calibration
            output_size: Size of output bird's eye view (width, height)
            calibration_scale: Pixels per cm for distance conversion
        """
        self.H = homography_matrix
        self.output_size = output_size
        self.calibration_scale = calibration_scale
        
        logger.info(f"PerspectiveTransformer initialized with output size {output_size}")
        logger.info(f"Calibration scale: {calibration_scale:.2f} pixels/cm")
    
    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Transform frame from camera view to bird's eye view
        
        Args:
            frame: Input frame in camera view (BGR)
            
        Returns:
            np.ndarray: Transformed frame in bird's eye view
        """
        warped = cv2.warpPerspective(frame, self.H, self.output_size)
        return warped
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform 2D points from camera view to bird's eye view
        
        Args:
            points: Points in camera view shape (N, 2) or (N, 1, 2)
            
        Returns:
            np.ndarray: Transformed points in bird's eye view
        """
        # Ensure points have correct shape
        original_shape = points.shape
        if len(points.shape) == 3:
            points = points.reshape(-1, 2)
        
        # Add homogeneous coordinate
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])
        
        # Transform
        transformed = points_homogeneous @ self.H.T
        
        # Normalize homogeneous coordinates
        transformed_2d = transformed[:, :2] / transformed[:, 2:3]
        
        # Restore original shape if needed
        if len(original_shape) == 3:
            transformed_2d = transformed_2d.reshape(original_shape)
        
        return transformed_2d
    
    def transform_lines(self, lines: np.ndarray) -> np.ndarray:
        """
        Transform lines (in Hough format) from camera view to bird's eye view
        
        Args:
            lines: Lines in Hough format [[rho, theta], ...]
            
        Returns:
            np.ndarray: Transformed lines in bird's eye view
        """
        # Convert Hough lines to point pairs
        point_pairs = []
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            point_pairs.append([[x1, y1], [x2, y2]])
        
        # Transform all points
        transformed_pairs = []
        for pair in point_pairs:
            p1 = self.transform_points(np.array([pair[0]]))[0]
            p2 = self.transform_points(np.array([pair[1]]))[0]
            transformed_pairs.append([p1, p2])
        
        return np.array(transformed_pairs)
    
    def get_roi_mask(self, roi_bottom_percent: float = 0.8) -> np.ndarray:
        """
        Get region of interest mask (bottom portion of bird's eye view)
        
        Args:
            roi_bottom_percent: Percentage of image height to use (0.0 to 1.0)
            
        Returns:
            np.ndarray: Binary mask
        """
        mask = np.zeros(self.output_size[::-1], dtype=np.uint8)
        
        # Calculate ROI
        height = self.output_size[1]
        roi_start_y = int(height * (1 - roi_bottom_percent))
        
        # Fill ROI
        mask[roi_start_y:, :] = 255
        
        return mask
    
    def pixel_to_cm(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to centimeters
        
        Args:
            pixel_distance: Distance in pixels
            
        Returns:
            float: Distance in cm
        """
        return pixel_distance / self.calibration_scale
    
    def cm_to_pixel(self, cm_distance: float) -> float:
        """
        Convert centimeter distance to pixels
        
        Args:
            cm_distance: Distance in cm
            
        Returns:
            float: Distance in pixels
        """
        return cm_distance * self.calibration_scale
    
    def apply_roi_mask(self, image: np.ndarray, roi_bottom_percent: float = 0.8) -> np.ndarray:
        """
        Apply ROI mask to image
        
        Args:
            image: Input image
            roi_bottom_percent: ROI percentage
            
        Returns:
            np.ndarray: Masked image
        """
        roi_mask = self.get_roi_mask(roi_bottom_percent)
        
        if len(image.shape) == 3:  # Color image
            roi_mask = np.stack([roi_mask] * 3, axis=2)
        
        return cv2.bitwise_and(image, image, mask=roi_mask)
