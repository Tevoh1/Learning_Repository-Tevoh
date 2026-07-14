# Video processing pipeline

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os

from lane_detection.calibration import CalibrationManager
from lane_detection.perspective_transform import PerspectiveTransformer
from lane_detection.lane_detection import HybridLaneDetector
from lane_detection.lane_tracking import LaneTracker, LaneAssociator, LanePolynomialFitter
from lane_detection.lane_analyzer import LaneAnalyzer
from lane_detection.utils import get_video_info, get_frame_sampling_indices, normalize_frame

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main pipeline for processing video and extracting lane measurements
    """
    
    def __init__(self,
                 calibration_manager: CalibrationManager,
                 config: Dict):
        """
        Initialize video processor
        
        Args:
            calibration_manager: Calibration manager instance
            config: Configuration dictionary
        """
        self.calibration_manager = calibration_manager
        self.config = config
        
        # Initialize components
        self.perspective_transformer = PerspectiveTransformer(
            calibration_manager.homography_matrix,
            calibration_manager.get_bird_eye_size(),
            calibration_manager.calibration_scale
        )
        
        self.detector = HybridLaneDetector(
            use_traditional=True,
            use_deep_learning=config.get('USE_DEEP_LEARNING', False),
            device=config.get('DEVICE_TYPE', 'cpu'),
            canny_low=config.get('CANNY_LOW_THRESHOLD', 50),
            canny_high=config.get('CANNY_HIGH_THRESHOLD', 150),
            hough_rho=config.get('HOUGH_RHO', 1),
            hough_theta=config.get('HOUGH_THETA', 1),
            hough_threshold=config.get('HOUGH_THRESHOLD', 50),
            min_line_length=config.get('HOUGH_MIN_LINE_LENGTH', 50),
            max_line_gap=config.get('HOUGH_MAX_LINE_GAP', 20)
        )
        
        self.lane_analyzer = LaneAnalyzer(
            calibration_manager.calibration_scale,
            config.get('MIN_LANE_WIDTH_CM', 200),
            config.get('MAX_LANE_WIDTH_CM', 600)
        )
        
        self.lane_tracker = LaneTracker(
            kalman_process_noise=config.get('KALMAN_PROCESS_NOISE', 0.1),
            kalman_measurement_noise=config.get('KALMAN_MEASUREMENT_NOISE', 5.0),
            temporal_smoothing_window=config.get('TEMPORAL_SMOOTHING_WINDOW', 5)
        )
        
        self.lane_associator = LaneAssociator(distance_threshold=50)
        
        self.lane_polynomial_fitter = LanePolynomialFitter(
            degree=config.get('POLYNOMIAL_DEGREE', 2),
            window_size=config.get('TEMPORAL_SMOOTHING_WINDOW', 5)
        )
        
        logger.info("VideoProcessor initialized")
    
    def process_video(self, video_path: str,
                     frame_sampling: int = 1,
                     save_frames_callback=None,
                     save_video_callback=None) -> List[Dict]:
        """
        Process video and extract lane measurements
        
        Args:
            video_path: Path to video file
            frame_sampling: Process every Nth frame
            save_frames_callback: Callback to save annotated frames
            save_video_callback: Callback to save annotated video
            
        Returns:
            List[Dict]: Frame measurements
        """
        logger.info(f"Processing video: {video_path}")
        
        # Get video info
        video_info = get_video_info(video_path)
        logger.info(f"Video info: {video_info}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Get frame sampling indices
        frame_indices = get_frame_sampling_indices(video_info['total_frames'], frame_sampling)
        logger.info(f"Will process {len(frame_indices)} frames")
        
        # Initialize output structures
        measurements = []
        video_frames_for_output = []  # For annotated video
        
        # Get ROI mask
        roi_mask = self.perspective_transformer.get_roi_mask(
            self.config.get('LANE_DETECTION_ROI_BOTTOM_PERCENT', 0.8)
        )
        
        # Process frames
        for frame_count, frame_index in enumerate(frame_indices):
            logger.info(f"Processing frame {frame_index}/{video_info['total_frames']}")
            
            # Read frame
            ret = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Cannot read frame {frame_index}")
                continue
            
            # Process frame
            frame_measurement = self._process_single_frame(
                frame, frame_index, video_info, frame_count
            )
            measurements.append(frame_measurement)
            
            # Optionally save annotated frame
            if save_frames_callback is not None:
                if frame_count % self.config.get('SAVE_FRAME_INTERVAL', 10) == 0:
                    annotated_original, annotated_bev = self._create_annotated_frames(
                        frame, frame_measurement
                    )
                    save_frames_callback(annotated_original, annotated_bev, frame_index)
            
            # Collect frame for video output
            if save_video_callback is not None:
                annotated_original, annotated_bev = self._create_annotated_frames(
                    frame, frame_measurement
                )
                video_frames_for_output.append((annotated_original, annotated_bev))
        
        cap.release()
        
        # Create and save video
        if save_video_callback is not None:
            save_video_callback(video_frames_for_output, video_info)
        
        logger.info(f"Video processing complete. Processed {len(measurements)} frames")
        
        return measurements
    
    def _process_single_frame(self, frame: np.ndarray,
                             frame_index: int,
                             video_info: Dict,
                             frame_count: int) -> Dict:
        """
        Process a single frame
        
        Args:
            frame: Input frame
            frame_index: Frame number in video
            video_info: Video information
            frame_count: Count of processed frames
            
        Returns:
            Dict: Frame measurement
        """
        measurement = {
            'frame_number': frame_index,
            'frame_count': frame_count,
            'timestamp_seconds': frame_index / video_info['fps'] if video_info['fps'] > 0 else 0,
            'video_name': os.path.basename(self.current_video_path),
        }
        
        # Normalize frame
        frame_normalized = normalize_frame(frame)
        
        # Transform to bird's eye view
        frame_bev = self.perspective_transformer.transform_frame(frame_normalized)
        
        # Get ROI mask
        roi_mask = self.perspective_transformer.get_roi_mask(
            self.config.get('LANE_DETECTION_ROI_BOTTOM_PERCENT', 0.8)
        )
        
        # Detect lanes
        detection_results = self.detector.detect(frame_bev, roi_mask)
        
        # Analyze lanes
        analysis_results = self.lane_analyzer.detect_lane_count_and_widths(
            detection_results['all_lines'],
            frame_bev.shape[0]
        )
        
        # Store raw detection
        measurement['num_lanes_raw'] = analysis_results['num_lanes']
        measurement['lane_widths_cm_raw'] = analysis_results['lane_widths_cm']
        measurement['lane_boundaries_raw'] = analysis_results['lane_boundaries']
        measurement['confidence_raw'] = analysis_results['confidence']
        measurement['detection_results'] = detection_results
        measurement['analysis_results'] = analysis_results
        
        # Track lanes and apply smoothing
        self._apply_tracking_and_smoothing(measurement)
        
        return measurement
    
    def _apply_tracking_and_smoothing(self, measurement: Dict):
        """
        Apply lane tracking and smoothing techniques
        
        Args:
            measurement: Frame measurement to update
        """
        # Get raw lane centers
        lane_centers = measurement['analysis_results']['lane_centers']
        
        # Initialize tracking results
        measurement['num_lanes_kalman'] = measurement['num_lanes_raw']
        measurement['num_lanes_temporal_avg'] = measurement['num_lanes_raw']
        measurement['num_lanes_polynomial'] = measurement['num_lanes_raw']
        
        measurement['lane_widths_cm_kalman'] = []
        measurement['lane_widths_cm_temporal_avg'] = []
        measurement['lane_widths_cm_polynomial'] = []
        
        measurement['confidence_kalman'] = measurement['confidence_raw']
        measurement['confidence_temporal_avg'] = measurement['confidence_raw']
        measurement['confidence_polynomial'] = measurement['confidence_raw']
        
        if len(lane_centers) == 0:
            return
        
        # Apply Kalman tracking
        kalman_widths = []
        for lane_id, width in enumerate(measurement['analysis_results']['lane_widths_cm']):
            tracking_result = self.lane_tracker.update_lane(lane_id, width)
            kalman_widths.append(tracking_result['kalman'])
            measurement['lane_widths_cm_kalman'].append(tracking_result['kalman'])
            measurement['lane_widths_cm_temporal_avg'].append(tracking_result['temporal_avg'])
        
        # Polynomial fitting (simplified)
        measurement['lane_widths_cm_polynomial'] = measurement['analysis_results']['lane_widths_cm']
    
    def _create_annotated_frames(self, original_frame: np.ndarray,
                                measurement: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create annotated frames for visualization
        
        Args:
            original_frame: Original video frame
            measurement: Frame measurement data
            
        Returns:
            Tuple: (annotated_original, annotated_bev)
        """
        # Transform to bird's eye view
        frame_normalized = normalize_frame(original_frame)
        frame_bev = self.perspective_transformer.transform_frame(frame_normalized)
        
        # Draw on bird's eye view
        annotated_bev = self._draw_lanes_on_bev(frame_bev.copy(), measurement)
        
        # Draw on original (simple annotation)
        annotated_original = original_frame.copy()
        
        # Add text with measurements
        cv2.putText(annotated_original,
                   f"Lanes: {measurement['num_lanes_raw']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   self.config.get('LANE_COUNT_TEXT_COLOR', (0, 255, 255)), 2)
        
        return annotated_original, annotated_bev
    
    def _draw_lanes_on_bev(self, frame_bev: np.ndarray, measurement: Dict) -> np.ndarray:
        """
        Draw detected lanes on bird's eye view
        
        Args:
            frame_bev: Bird's eye view frame
            measurement: Frame measurement data
            
        Returns:
            np.ndarray: Annotated frame
        """
        # Draw lane boundaries
        boundaries = measurement['analysis_results']['lane_boundaries']
        if len(boundaries) >= 2:
            for i in range(len(boundaries) - 1):
                x1 = int(boundaries[i])
                x2 = int(boundaries[i + 1])
                center = (x1 + x2) // 2
                
                # Draw boundaries
                cv2.line(frame_bev, (x1, 0), (x1, frame_bev.shape[0]),
                        self.config.get('LANE_COLOR_PRIMARY', (0, 255, 0)), 2)
                if i == len(boundaries) - 2:  # Last boundary
                    cv2.line(frame_bev, (x2, 0), (x2, frame_bev.shape[0]),
                            self.config.get('LANE_COLOR_PRIMARY', (0, 255, 0)), 2)
                
                # Draw lane center
                cv2.line(frame_bev, (center, 0), (center, frame_bev.shape[0]),
                        self.config.get('LANE_CENTER_COLOR', (255, 0, 0)), 1)
                
                # Draw lane width label
                if i < len(measurement['analysis_results']['lane_widths_cm']):
                    width_cm = measurement['analysis_results']['lane_widths_cm'][i]
                    cv2.putText(frame_bev,
                               f"{width_cm:.1f}cm",
                               (center - 30, 50 + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                               self.config.get('LANE_WIDTH_TEXT_COLOR', (255, 255, 255)), 1)
        
        # Draw lane count
        cv2.putText(frame_bev,
                   f"Lanes: {measurement['num_lanes_raw']}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                   self.config.get('LANE_COUNT_TEXT_COLOR', (0, 255, 255)), 2)
        
        return frame_bev
    
    def set_current_video_path(self, video_path: str):
        """
        Set the current video path for reference
        
        Args:
            video_path: Path to current video
        """
        self.current_video_path = video_path
