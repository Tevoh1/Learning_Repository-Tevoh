# Main entry point for lane detection system

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import configuration
import config
from lane_detection.calibration import CalibrationManager
from lane_detection.video_processor import VideoProcessor
from lane_detection.output_handler import OutputHandler
from lane_detection.utils import setup_device, create_output_directories


def main():
    """
    Main entry point for lane detection pipeline
    """
    logger.info("="*80)
    logger.info("LANE DETECTION SYSTEM - MAIN PIPELINE")
    logger.info("="*80)
    
    # Setup device
    if config.USE_CUDA_IF_AVAILABLE:
        device = setup_device()
        config.DEVICE_TYPE = device
    else:
        config.DEVICE_TYPE = 'cpu'
        logger.info("Using CPU (as configured)")
    
    # Create output directories
    create_output_directories(config.OUTPUT_FOLDER)
    
    # Initialize calibration
    logger.info("\n" + "="*80)
    logger.info("STEP 1: CALIBRATION")
    logger.info("="*80)
    
    try:
        calibration_manager = CalibrationManager(
            config.CALIBRATION_IMAGE_PATH,
            sheet_size_cm=config.SHEET_SIZE_CM
        )
        
        homography, calibration_scale = calibration_manager.calibrate(
            use_auto_detect=config.AUTO_DETECT_WHITE_SHEET,
            manual_confirm=config.MANUAL_CONFIRMATION
        )
        
        logger.info(f"Calibration successful!")
        logger.info(f"Calibration scale: {calibration_scale:.2f} pixels/cm")
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False
    
    # Initialize output handler
    output_handler = OutputHandler(config.OUTPUT_FOLDER, config.__dict__)
    
    # Initialize video processor
    logger.info("\n" + "="*80)
    logger.info("STEP 2: VIDEO PROCESSING")
    logger.info("="*80)
    
    video_processor = VideoProcessor(calibration_manager, config.__dict__)
    
    # Get list of videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = []
    
    if os.path.isdir(config.INPUT_VIDEO_FOLDER):
        for file in os.listdir(config.INPUT_VIDEO_FOLDER):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                videos.append(os.path.join(config.INPUT_VIDEO_FOLDER, file))
    else:
        logger.error(f"Input folder not found: {config.INPUT_VIDEO_FOLDER}")
        return False
    
    if len(videos) == 0:
        logger.error(f"No videos found in {config.INPUT_VIDEO_FOLDER}")
        return False
    
    logger.info(f"Found {len(videos)} video(s) to process")
    
    # Process each video
    for video_idx, video_path in enumerate(videos, 1):
        logger.info(f"\n[{video_idx}/{len(videos)}] Processing: {os.path.basename(video_path)}")
        
        try:
            video_processor.set_current_video_path(video_path)
            
            # Define callbacks for saving frames and videos
            def save_frames_callback(orig_frame, bev_frame, frame_num):
                output_handler.save_annotated_frames(
                    orig_frame, bev_frame, frame_num,
                    os.path.basename(video_path)
                )
            
            def save_video_callback(frames, video_info):
                if config.OUTPUT_ANNOTATED_VIDEO:
                    output_handler.save_annotated_video(
                        frames, video_info,
                        os.path.basename(video_path)
                    )
            
            # Process video
            measurements = video_processor.process_video(
                video_path,
                frame_sampling=config.FRAME_SAMPLING,
                save_frames_callback=save_frames_callback if config.OUTPUT_SAMPLE_FRAMES else None,
                save_video_callback=save_video_callback
            )
            
            # Add measurements to output handler
            output_handler.add_measurements(measurements, os.path.basename(video_path))
            
            logger.info(f"✓ Completed: {os.path.basename(video_path)}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {os.path.basename(video_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save CSV output
    logger.info("\n" + "="*80)
    logger.info("STEP 3: SAVING RESULTS")
    logger.info("="*80)
    
    output_handler.save_csv()
    logger.info(f"✓ CSV saved to: {config.CSV_OUTPUT_PATH}")
    
    # Generate statistics
    if len(output_handler.all_measurements) > 0:
        stats = output_handler.generate_statistics_report(output_handler.all_measurements)
        logger.info("\nStatistics Report:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value:.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Results saved to: {config.OUTPUT_FOLDER}")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
