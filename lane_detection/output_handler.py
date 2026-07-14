# Output handling module - CSV writing and frame/video saving

import cv2
import numpy as np
import pandas as pd
import logging
import os
from typing import List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class OutputHandler:
    """
    Handles all output operations: CSV, annotated frames, and videos
    """
    
    def __init__(self, output_folder: str, config: Dict):
        """
        Initialize output handler
        
        Args:
            output_folder: Root output folder
            config: Configuration dictionary
        """
        self.output_folder = output_folder
        self.config = config
        
        # Create subdirectories
        self.frames_folder = os.path.join(output_folder, 'annotated_frames')
        self.videos_folder = os.path.join(output_folder, 'annotated_videos')
        
        os.makedirs(self.frames_folder, exist_ok=True)
        os.makedirs(self.videos_folder, exist_ok=True)
        
        # CSV output path
        self.csv_path = config.get('CSV_OUTPUT_PATH',
                                   os.path.join(output_folder, 'lane_measurements.csv'))
        
        # Initialize dataframe for accumulating results
        self.all_measurements = []
        
        logger.info(f"OutputHandler initialized")
        logger.info(f"Output folder: {output_folder}")
        logger.info(f"CSV path: {self.csv_path}")
    
    def add_measurements(self, measurements: List[Dict], video_name: str):
        """
        Add frame measurements to output buffer
        
        Args:
            measurements: List of frame measurements
            video_name: Name of video being processed
        """
        for measurement in measurements:
            output_row = self._prepare_csv_row(measurement, video_name)
            self.all_measurements.append(output_row)
        
        logger.info(f"Added {len(measurements)} measurements for {video_name}")
    
    def _prepare_csv_row(self, measurement: Dict, video_name: str) -> Dict:
        """
        Prepare a single row for CSV output
        
        Args:
            measurement: Frame measurement
            video_name: Name of video
            
        Returns:
            Dict: Row data for CSV
        """
        row = {
            'video_name': video_name,
            'frame_number': measurement.get('frame_number', 0),
            'timestamp_seconds': measurement.get('timestamp_seconds', 0.0),
            
            # Raw detections
            'num_lanes_raw': measurement.get('num_lanes_raw', 0),
            'confidence_raw': measurement.get('confidence_raw', 0.0),
            
            # Kalman filter results
            'num_lanes_kalman': measurement.get('num_lanes_kalman', 0),
            'confidence_kalman': measurement.get('confidence_kalman', 0.0),
            
            # Temporal smoothing results
            'num_lanes_temporal_avg': measurement.get('num_lanes_temporal_avg', 0),
            'confidence_temporal_avg': measurement.get('confidence_temporal_avg', 0.0),
            
            # Polynomial fitting results
            'num_lanes_polynomial': measurement.get('num_lanes_polynomial', 0),
            'confidence_polynomial': measurement.get('confidence_polynomial', 0.0),
        }
        
        # Add lane widths (raw)
        lane_widths_cm_raw = measurement.get('lane_widths_cm_raw', [])
        for i, width in enumerate(lane_widths_cm_raw):
            row[f'lane_width_{i+1}_cm_raw'] = width
        
        # Add lane widths (kalman)
        lane_widths_cm_kalman = measurement.get('lane_widths_cm_kalman', [])
        for i, width in enumerate(lane_widths_cm_kalman):
            row[f'lane_width_{i+1}_cm_kalman'] = width
        
        # Add lane widths (temporal)
        lane_widths_cm_temporal = measurement.get('lane_widths_cm_temporal_avg', [])
        for i, width in enumerate(lane_widths_cm_temporal):
            row[f'lane_width_{i+1}_cm_temporal_avg'] = width
        
        # Add lane widths (polynomial)
        lane_widths_cm_poly = measurement.get('lane_widths_cm_polynomial', [])
        for i, width in enumerate(lane_widths_cm_poly):
            row[f'lane_width_{i+1}_cm_polynomial'] = width
        
        return row
    
    def save_csv(self):
        """
        Save all measurements to CSV
        """
        if len(self.all_measurements) == 0:
            logger.warning("No measurements to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.all_measurements)
        
        # Fill NaN values with empty strings for better readability
        df = df.fillna('')
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        logger.info(f"CSV saved to {self.csv_path}")
        logger.info(f"Total rows: {len(df)}")
    
    def save_annotated_frames(self, original_frame: np.ndarray,
                             bev_frame: np.ndarray,
                             frame_number: int,
                             video_name: str):
        """
        Save annotated frames (side-by-side)
        
        Args:
            original_frame: Original view
            bev_frame: Bird's eye view
            frame_number: Frame number
            video_name: Video name (for file naming)
        """
        # Create side-by-side image
        # Resize BEV to match original height
        original_height = original_frame.shape[0]
        bev_resized = cv2.resize(bev_frame, (bev_frame.shape[1], original_height))
        
        # Concatenate horizontally
        side_by_side = np.hstack([original_frame, bev_resized])
        
        # Create filename
        video_name_clean = video_name.replace('.mp4', '').replace('.avi', '')
        filename = f"{video_name_clean}_frame_{frame_number:06d}.png"
        filepath = os.path.join(self.frames_folder, filename)
        
        # Save
        cv2.imwrite(filepath, side_by_side)
        logger.debug(f"Saved annotated frame: {filepath}")
    
    def save_annotated_video(self, frames: List[Tuple[np.ndarray, np.ndarray]],
                            video_info: Dict,
                            video_name: str):
        """
        Save annotated video with side-by-side views
        
        Args:
            frames: List of (original, bev) frame tuples
            video_info: Video information
            video_name: Video name (for file naming)
        """
        if len(frames) == 0:
            logger.warning("No frames to save for video")
            return
        
        # Get frame dimensions
        original_frame = frames[0][0]
        bev_frame = frames[0][1]
        
        # Resize BEV to match original height
        original_height = original_frame.shape[0]
        bev_width = int(bev_frame.shape[1] * original_height / bev_frame.shape[0])
        bev_resized = cv2.resize(bev_frame, (bev_width, original_height))
        
        # Output video dimensions
        output_width = original_frame.shape[1] + bev_resized.shape[1]
        output_height = original_height
        
        # Create video writer
        video_name_clean = video_name.replace('.mp4', '').replace('.avi', '')
        output_filename = f"{video_name_clean}_annotated.mp4"
        output_path = os.path.join(self.videos_folder, output_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config.get('ANNOTATED_VIDEO_FPS', 30)
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        if not out.isOpened():
            logger.error(f"Cannot create video writer for {output_path}")
            return
        
        # Write frames
        for original, bev in frames:
            # Resize BEV
            bev_resized = cv2.resize(bev, (bev_width, original_height))
            
            # Create side-by-side
            side_by_side = np.hstack([original, bev_resized])
            
            # Write frame
            out.write(side_by_side)
        
        out.release()
        logger.info(f"Saved annotated video: {output_path}")
    
    def generate_statistics_report(self, measurements: List[Dict]) -> Dict:
        """
        Generate statistics report from measurements
        
        Args:
            measurements: List of frame measurements
            
        Returns:
            Dict: Statistics report
        """
        if len(measurements) == 0:
            return {}
        
        report = {}
        
        # Number of lanes statistics
        num_lanes_raw = [m.get('num_lanes_raw', 0) for m in measurements]
        report['avg_lanes_raw'] = np.mean(num_lanes_raw)
        report['std_lanes_raw'] = np.std(num_lanes_raw)
        report['mode_lanes_raw'] = int(np.median(num_lanes_raw))
        
        # Confidence statistics
        conf_raw = [m.get('confidence_raw', 0.0) for m in measurements]
        report['avg_confidence_raw'] = np.mean(conf_raw)
        report['min_confidence_raw'] = np.min(conf_raw)
        report['max_confidence_raw'] = np.max(conf_raw)
        
        # Lane width statistics
        all_widths_cm = []
        for m in measurements:
            widths = m.get('lane_widths_cm_raw', [])
            all_widths_cm.extend(widths)
        
        if all_widths_cm:
            report['avg_lane_width_cm'] = np.mean(all_widths_cm)
            report['std_lane_width_cm'] = np.std(all_widths_cm)
            report['min_lane_width_cm'] = np.min(all_widths_cm)
            report['max_lane_width_cm'] = np.max(all_widths_cm)
        
        return report
