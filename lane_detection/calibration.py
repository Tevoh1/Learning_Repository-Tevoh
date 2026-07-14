# Calibration module for perspective transform setup

import cv2
import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CalibrationManager:
    """
    Manages calibration using white sheet image.
    Detects white sheet corners and computes homography matrix.
    """
    
    def __init__(self, calibration_image_path: str, sheet_size_cm: float = 200):
        """
        Initialize calibration manager
        
        Args:
            calibration_image_path: Path to calibration image with white sheet
            sheet_size_cm: Size of white sheet in cm (default 200cm = 2m)
        """
        self.calibration_image_path = calibration_image_path
        self.sheet_size_cm = sheet_size_cm
        self.image = cv2.imread(calibration_image_path)
        
        if self.image is None:
            raise RuntimeError(f"Cannot load calibration image: {calibration_image_path}")
        
        self.image_height, self.image_width = self.image.shape[:2]
        self.homography_matrix = None
        self.calibration_scale = None  # pixels per cm
        self.sheet_corners = None  # Detected corners in image space
        
        logger.info(f"Calibration image loaded: {self.image_width}x{self.image_height}")
    
    def auto_detect_white_sheet(self, 
                                hsv_lower: Tuple[int, int, int] = (0, 0, 200),
                                hsv_upper: Tuple[int, int, int] = (180, 30, 255)) -> Optional[np.ndarray]:
        """
        Automatically detect white sheet in calibration image
        
        Args:
            hsv_lower: Lower HSV threshold for white
            hsv_upper: Upper HSV threshold for white
            
        Returns:
            np.ndarray: 4 corner points of detected sheet, or None if failed
        """
        logger.info("Attempting automatic white sheet detection...")
        
        # Convert to HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Create mask for white color
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No white regions detected")
            return None
        
        # Find largest contour (should be the white sheet)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        logger.info(f"Detected region area: {area} pixels")
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) != 4:
            logger.warning(f"Expected 4 corners, got {len(approx)}")
            # Try convex hull
            hull = cv2.convexHull(largest_contour)
            if len(hull) == 4:
                approx = hull
            else:
                return None
        
        corners = np.float32([point[0] for point in approx])
        logger.info(f"Auto-detected corners: {corners}")
        
        return corners
    
    def manual_select_corners(self) -> np.ndarray:
        """
        Manually select corners by clicking on the image.
        Click the 4 corners of the white sheet in order: top-left, top-right, bottom-right, bottom-left
        
        Returns:
            np.ndarray: 4 corner points
        """
        logger.info("Opening image for manual corner selection...")
        logger.info("Click on corners in order: top-left, top-right, bottom-right, bottom-left")
        
        corners = []
        image_copy = self.image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                corners.append([x, y])
                cv2.circle(image_copy, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(image_copy, str(len(corners)), (x + 10, y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Select Calibration Corners', image_copy)
                logger.info(f"Corner {len(corners)} selected at ({x}, {y})")
        
        cv2.imshow('Select Calibration Corners', image_copy)
        cv2.setMouseCallback('Select Calibration Corners', mouse_callback)
        
        logger.info("Waiting for 4 clicks (press any key when done)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if len(corners) != 4:
            logger.error(f"Expected 4 corners, got {len(corners)}")
            raise ValueError("Must select exactly 4 corners")
        
        return np.float32(corners)
    
    def calibrate(self, use_auto_detect: bool = True, manual_confirm: bool = False) -> Tuple[np.ndarray, float]:
        """
        Calibrate using white sheet. Computes homography and scale.
        
        Args:
            use_auto_detect: Try automatic detection first
            manual_confirm: Allow manual correction if auto-detect fails or for confirmation
            
        Returns:
            Tuple: (homography_matrix, calibration_scale in pixels/cm)
        """
        corners = None
        
        # Try automatic detection
        if use_auto_detect:
            corners = self.auto_detect_white_sheet()
            
            if corners is not None and manual_confirm:
                logger.info("Auto-detection successful. Confirming with manual selection...")
                self._show_detected_corners(corners)
                user_confirm = input("Are these corners correct? (y/n): ").lower()
                if user_confirm != 'y':
                    logger.info("Using manual selection instead...")
                    corners = self.manual_select_corners()
        
        # If auto-detect failed, use manual
        if corners is None:
            logger.info("Using manual corner selection...")
            corners = self.manual_select_corners()
        
        self.sheet_corners = corners
        
        # Compute homography
        # Image coordinates (source)
        src_points = corners
        
        # World coordinates (destination) - bird's eye view
        # Assume sheet is 200cm x 200cm, place at reasonable position
        dst_width = 800  # pixels in output
        dst_height = 800  # pixels in output
        padding = 50
        
        dst_points = np.float32([
            [padding, padding],  # top-left
            [dst_width - padding, padding],  # top-right
            [dst_width - padding, dst_height - padding],  # bottom-right
            [padding, dst_height - padding]  # bottom-left
        ])
        
        # Compute homography
        self.homography_matrix, status = cv2.findHomography(src_points, dst_points)
        
        if self.homography_matrix is None:
            raise RuntimeError("Failed to compute homography matrix")
        
        logger.info("Homography matrix computed successfully")
        
        # Compute calibration scale
        # Distance between corners in output (bird's eye)
        top_side_pixels = np.linalg.norm(dst_points[1] - dst_points[0])
        # This corresponds to sheet_size_cm
        self.calibration_scale = top_side_pixels / self.sheet_size_cm
        
        logger.info(f"Calibration scale: {self.calibration_scale:.2f} pixels/cm")
        
        # Validate calibration
        self._validate_calibration()
        
        return self.homography_matrix, self.calibration_scale
    
    def _show_detected_corners(self, corners: np.ndarray):
        """
        Display image with detected corners
        
        Args:
            corners: Corner points to display
        """
        display_image = self.image.copy()
        corners = np.int32(corners)
        
        for i, corner in enumerate(corners):
            cv2.circle(display_image, tuple(corner), 8, (0, 255, 0), -1)
            cv2.putText(display_image, str(i + 1), tuple(corner + [15, 15]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw polygon
        cv2.polylines(display_image, [corners], True, (0, 255, 0), 2)
        
        cv2.imshow('Detected Corners', display_image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
    
    def _validate_calibration(self):
        """
        Validate calibration by transforming the white sheet
        """
        # Transform the calibration image
        output_size = (800, 800)
        warped = cv2.warpPerspective(self.image, self.homography_matrix, output_size)
        
        logger.info("Calibration validation: Computing white area in transformed image...")
        
        # Check that white area is present in transformed image
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_area = cv2.countNonZero(mask)
        
        if white_area < 10000:  # At least some white region
            logger.warning(f"Low white area detected in transformed image: {white_area} pixels")
        else:
            logger.info(f"White area in transformed image: {white_area} pixels (validation passed)")
    
    def get_bird_eye_size(self) -> Tuple[int, int]:
        """
        Get the size of bird's eye view output
        
        Returns:
            Tuple: (width, height) in pixels
        """
        return (800, 800)  # Should match dst_width and dst_height above
    
    def save_calibration(self, output_path: str):
        """
        Save calibration data to file
        
        Args:
            output_path: Path to save calibration file (.npz)
        """
        np.savez(output_path,
                homography_matrix=self.homography_matrix,
                calibration_scale=self.calibration_scale,
                sheet_corners=self.sheet_corners)
        logger.info(f"Calibration saved to {output_path}")
    
    def load_calibration(self, input_path: str):
        """
        Load calibration data from file
        
        Args:
            input_path: Path to calibration file (.npz)
        """
        data = np.load(input_path)
        self.homography_matrix = data['homography_matrix']
        self.calibration_scale = float(data['calibration_scale'])
        self.sheet_corners = data['sheet_corners']
        logger.info(f"Calibration loaded from {input_path}")
