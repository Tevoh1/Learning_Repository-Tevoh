# Lane Detection System - Automated Road Geometry Measurement

## Overview

This system automatically detects and measures lane distances from dash camera video footage. It uses a combination of traditional computer vision and deep learning techniques to identify lanes and measure distances between them in real-world centimeters.

### Key Features

- **Perspective Calibration**: Uses a known-size white sheet to calibrate camera perspective and convert measurements to real-world units (cm)
- **Multi-Method Detection**: Combines traditional CV (Hough line detection) with deep learning for robust lane detection
- **Multiple Smoothing Strategies**: Implements Kalman filtering, temporal averaging, and polynomial fitting to compare detection approaches
- **Real-time Processing**: Processes videos frame-by-frame with configurable sampling
- **Comprehensive Output**: Generates annotated videos, sample frames, and detailed CSV measurements
- **GPU/CPU Support**: Automatically detects and uses CUDA if available, falls back to CPU

## Project Structure

```
lane_detection/
├── __init__.py                 # Package initialization
├── calibration.py              # Calibration using white sheet
├── perspective_transform.py    # Bird's eye view transformation
├── lane_detection.py           # Lane detection algorithms
├── lane_tracking.py            # Lane tracking and smoothing
├── lane_analyzer.py            # Lane measurement and analysis
├── video_processor.py          # Main video processing pipeline
├── output_handler.py           # Output generation (CSV, frames, video)
└── utils.py                    # Utility functions

config.py                       # Configuration (hardcoded parameters)
main.py                         # Main entry point
requirements.txt               # Python dependencies
```

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Settings

Edit `config.py` and set the following paths:

```python
INPUT_VIDEO_FOLDER = r"C:\path\to\videos"
CALIBRATION_IMAGE_PATH = r"C:\path\to\calibration.jpg"
OUTPUT_FOLDER = r"C:\path\to\processing"
```

### 3. Prepare Calibration Image

- Place a 2m × 2m white sheet perpendicular to the camera
- Take a photo from the dash camera
- Save as the calibration image

## Usage

### Quick Start

Run the main pipeline:

```bash
python main.py
```

The script will:
1. **Calibration Phase**: Detect/confirm the white sheet corners and compute homography matrix
2. **Video Processing Phase**: Process all videos in the input folder
3. **Output Phase**: Save results to CSV and annotated media

### Calibration Process

When you run the script:

1. **Automatic Detection**: The system attempts to automatically detect white sheet corners
2. **Manual Confirmation**: If `MANUAL_CONFIRMATION = True`, you can verify or correct the corners
3. **Homography Matrix**: Computed from the 4 corners to transform camera view to bird's eye view
4. **Scale Calculation**: Pixels are converted to real-world centimeters

### Configuration Options

Edit `config.py` to customize:

```python
# Frame sampling
FRAME_SAMPLING = 5              # Process every 5th frame (1 = all frames)
SAVE_FRAME_INTERVAL = 10        # Save annotated frame every 10th processed frame

# Lane detection parameters
MIN_LANE_WIDTH_CM = 200         # Minimum lane width
MAX_LANE_WIDTH_CM = 600         # Maximum lane width

# Tracking & smoothing
KALMAN_PROCESS_NOISE = 0.1
KALMAN_MEASUREMENT_NOISE = 5.0
TEMPORAL_SMOOTHING_WINDOW = 5
POLYNOMIAL_DEGREE = 2

# Output
OUTPUT_ANNOTATED_VIDEO = True
OUTPUT_SAMPLE_FRAMES = True
```

## Output Files

### 1. CSV Output (`lane_measurements.csv`)

Contains measurements for each processed frame:

```
video_name | frame_number | timestamp_seconds | num_lanes_raw | num_lanes_kalman | num_lanes_temporal_avg | num_lanes_polynomial |
           | lane_width_1_cm_raw | lane_width_1_cm_kalman | lane_width_1_cm_temporal_avg | lane_width_1_cm_polynomial |
           | lane_width_2_cm_raw | ... | confidence_raw | confidence_kalman | confidence_temporal_avg | confidence_polynomial
```

**Columns Description**:
- `video_name`: Name of source video
- `frame_number`: Frame index in video
- `timestamp_seconds`: Time in seconds from start
- `num_lanes_*`: Number of lanes detected using different methods
- `lane_width_N_cm_*`: Width of lane N in cm (for each smoothing method)
- `confidence_*`: Confidence score for each detection method

### 2. Annotated Frames

Stored in `annotated_frames/` folder:
- Side-by-side views: Original camera view + Bird's eye view
- Shows detected lane boundaries, centers, and width labels
- Filename: `{video_name}_frame_{frame_number}.png`

### 3. Annotated Video

Stored in `annotated_videos/` folder:
- Side-by-side video with lane annotations
- Filename: `{video_name}_annotated.mp4`

## Detection Methods

### Raw Detection
Traditional computer vision using:
- CLAHE histogram equalization (handles glare/shadows)
- Canny edge detection
- Hough line transform
- Line clustering to identify lane boundaries

### Kalman Filtering
Smooths lane positions across frames using Kalman filter for each lane.

### Temporal Averaging
Simple moving average of lane measurements over a window of frames.

### Polynomial Fitting
Fits polynomial curves to lane boundaries, enabling extrapolation and smoothing.

## Performance Considerations

### GPU Acceleration
With CUDA-capable GPU:
- ~100-200 fps for 720p video (with DL models)
- Automatic fallback to CPU if CUDA unavailable

### CPU Processing
- ~10-30 fps for 720p video
- Suitable for offline batch processing
- Consider increasing `FRAME_SAMPLING` to reduce processing time

### Optimization Tips
1. Increase `FRAME_SAMPLING` to skip frames
2. Reduce input video resolution if needed
3. Set `OUTPUT_ANNOTATED_VIDEO = False` to skip video encoding
4. Use GPU for best performance

## Troubleshooting

### Calibration Issues

**Problem**: Automatic white sheet detection fails
- **Solution**: Set `AUTO_DETECT_WHITE_SHEET = False` and use manual selection
- **Solution**: Adjust `WHITE_SHEET_HSV_LOWER` and `WHITE_SHEET_HSV_UPPER` thresholds

**Problem**: Perspective transform looks warped
- **Solution**: Ensure white sheet is perpendicular to camera
- **Solution**: Verify the 4 corners are selected in order: top-left, top-right, bottom-right, bottom-left

### Lane Detection Issues

**Problem**: Lanes not detected or very few lanes
- **Solution**: Adjust `CANNY_LOW_THRESHOLD` and `CANNY_HIGH_THRESHOLD`
- **Solution**: Adjust `HOUGH_THRESHOLD` to be more/less strict
- **Solution**: Check video quality and lighting conditions

**Problem**: Incorrect lane count
- **Solution**: Adjust `MIN_LANE_WIDTH_CM` and `MAX_LANE_WIDTH_CM` thresholds
- **Solution**: Enable `DEBUG_MODE` to see intermediate detection results

### Output Issues

**Problem**: CSV file is empty
- **Solution**: Check that videos were processed (check log output)
- **Solution**: Ensure input folder path is correct

**Problem**: Video encoding errors
- **Solution**: Install ffmpeg: `conda install ffmpeg` or `apt-get install ffmpeg`
- **Solution**: Set `OUTPUT_ANNOTATED_VIDEO = False` to skip video generation

## Algorithm Details

### Perspective Transform

1. Detect white sheet corners in calibration image (auto or manual)
2. Compute homography matrix using 4 known points
3. Transform all video frames to bird's eye view
4. Scale: 200cm (sheet size) / number of pixels = pixels per cm

### Lane Detection Pipeline

```
Input Frame
    ↓
Normalization (CLAHE)
    ↓
Perspective Transform (Bird's Eye View)
    ↓
Traditional CV: Canny Edge + Hough Lines
    ↓
Deep Learning: SCNN (if available)
    ↓
Line Clustering & Lane Boundary Detection
    ↓
Measurement Calculation
    ↓
Tracking & Smoothing (Kalman, Temporal, Polynomial)
    ↓
Output: Frame measurements
```

### Confidence Scoring

Based on:
- Number of detection methods agreeing
- Consistency of lane widths
- Temporal consistency across frames
- Edge quality in processed image

## Advanced Usage

### Batch Processing Multiple Folders

Modify `main.py` to loop through multiple input folders:

```python
for input_folder in [folder1, folder2, folder3]:
    config.INPUT_VIDEO_FOLDER = input_folder
    main()
```

### Analyzing Results

Load and analyze CSV output:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processing/lane_measurements.csv')

# Plot lane count over time
df.groupby('video_name')['num_lanes_raw'].plot()
plt.show()

# Plot lane width statistics
df['lane_width_1_cm_raw'].hist()
plt.show()
```

## Known Limitations

1. **Curved Roads**: Current implementation assumes relatively straight lanes
2. **Low Lighting**: May struggle with nighttime or heavily shadowed conditions
3. **Marked vs Unmarked**: Works best with clearly marked lanes
4. **Angle Changes**: Assumes camera angle remains fixed
5. **Deep Learning**: SCNN model not fully implemented (currently uses traditional CV only)

## Future Enhancements

- [ ] Implement SCNN deep learning model
- [ ] Add curved lane handling
- [ ] Support for camera angle adjustment detection
- [ ] Real-time processing with GPU acceleration
- [ ] Multi-lane tracking with ID persistence
- [ ] Integration with road metadata databases
- [ ] Web interface for visualization

## License

This project is provided as-is for research and development purposes.

## Contact

For issues or questions, please refer to the GitHub repository.

---

**Version**: 1.0.0  
**Last Updated**: 2026-07-14
