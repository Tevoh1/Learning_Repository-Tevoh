# Testing and Debugging Guide

## Overview

This guide provides comprehensive information on testing the lane detection system, debugging issues, and validating output quality.

## Unit Testing

### Testing Individual Modules

```python
# test_calibration.py
import cv2
from lane_detection.calibration import CalibrationManager

# Test calibration loading
cal_manager = CalibrationManager('calibration.jpg')

# Test auto-detection
corners = cal_manager.auto_detect_white_sheet()
print(f"Detected corners: {corners}")

# Test calibration
H, scale = cal_manager.calibrate(use_auto_detect=True, manual_confirm=False)
print(f"Homography matrix:\n{H}")
print(f"Scale: {scale} pixels/cm")
```

```python
# test_perspective.py
from lane_detection.perspective_transform import PerspectiveTransformer
import numpy as np

# Initialize with mock homography
H = np.eye(3)
transformer = PerspectiveTransformer(H, (800, 800), 2.0)

# Test point transformation
points = np.array([[100, 50], [200, 100]])
transformed = transformer.transform_points(points)
print(f"Transformed points:\n{transformed}")

# Test pixel to cm conversion
pixels = 100
cm = transformer.pixel_to_cm(pixels)
print(f"{pixels} pixels = {cm} cm")
```

```python
# test_lane_detection.py
from lane_detection.lane_detection import TraditionalLaneDetector
import cv2
import numpy as np

# Load test image
image = cv2.imread('test_frame.jpg')

# Create detector
detector = TraditionalLaneDetector()

# Detect lanes
lines, edges = detector.detect(image)
print(f"Detected {len(lines)} lines")

# Visualize
img_lines = image.copy()
for line in lines:
    x1, y1, x2, y2 = line
    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detected Lines', img_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Integration Testing

### Test Full Pipeline on Single Video

```python
# test_pipeline.py
import config
from lane_detection.calibration import CalibrationManager
from lane_detection.video_processor import VideoProcessor
from lane_detection.output_handler import OutputHandler

# Setup
cal_manager = CalibrationManager(config.CALIBRATION_IMAGE_PATH)
H, scale = cal_manager.calibrate()

processor = VideoProcessor(cal_manager, config.__dict__)
output_handler = OutputHandler(config.OUTPUT_FOLDER, config.__dict__)

# Process single video
test_video = 'test_video.mp4'
measurements = processor.process_video(test_video, frame_sampling=5)

# Save results
output_handler.add_measurements(measurements, 'test_video.mp4')
output_handler.save_csv()

print(f"Processed {len(measurements)} frames")
print(f"Results saved to {config.CSV_OUTPUT_PATH}")
```

## Validation Testing

### Validate Calibration Quality

```python
# validate_calibration.py
import cv2
import numpy as np
from lane_detection.calibration import CalibrationManager

cal_manager = CalibrationManager('calibration.jpg')
H, scale = cal_manager.calibrate()

# Test image transformation
original = cv2.imread('calibration.jpg')
transformed = cv2.warpPerspective(original, H, (800, 800))

# Check that white sheet is rectangular in transformed space
hsv = cv2.cvtColor(transformed, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))

# Calculate aspect ratio
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h > 0 else 0
    print(f"Aspect ratio: {aspect_ratio:.2f} (should be close to 1.0)")
    
    # Visualize
    cv2.imshow('Original', original)
    cv2.imshow('Transformed', transformed)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

### Validate Lane Detection

```python
# validate_lane_detection.py
import cv2
import numpy as np
from lane_detection.lane_detection import HybridLaneDetector
from lane_detection.lane_analyzer import LaneAnalyzer

# Load bird's eye view test image
image_bev = cv2.imread('test_bev_frame.jpg')

# Detect lanes
detector = HybridLaneDetector()
results = detector.detect(image_bev)

print(f"CV lines detected: {len(results['cv_lines'])}")
print(f"DL lines detected: {len(results['dl_lines'])}")
print(f"Confidence: {results['confidence']:.2f}")

# Analyze lanes
analyzer = LaneAnalyzer(2.0, 200, 600)  # scale, min_width, max_width
analysis = analyzer.detect_lane_count_and_widths(results['all_lines'], image_bev.shape[0])

print(f"Number of lanes: {analysis['num_lanes']}")
print(f"Lane widths (cm): {analysis['lane_widths_cm']}")
print(f"Confidence: {analysis['confidence']:.2f}")

# Visualize
img_annotated = image_bev.copy()
for boundary in analysis['lane_boundaries']:
    x = int(boundary)
    cv2.line(img_annotated, (x, 0), (x, img_annotated.shape[0]), (0, 255, 0), 2)

cv2.imshow('Lane Detection', img_annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Debugging Strategies

### Enable Debug Mode

```python
# config.py
DEBUG_MODE = True
SAVE_INTERMEDIATE_IMAGES = True
```

This will save intermediate processing results:
- Bird's eye view frames
- Edge detection results
- Hough line detection results

### Visual Debugging

```python
# debug_frame.py
import cv2
import numpy as np
from lane_detection.calibration import CalibrationManager
from lane_detection.perspective_transform import PerspectiveTransformer
from lane_detection.lane_detection import HybridLaneDetector
from lane_detection.utils import normalize_frame

# Setup
cal_manager = CalibrationManager('calibration.jpg')
H, scale = cal_manager.calibrate()
transformer = PerspectiveTransformer(H, (800, 800), scale)

# Load and process frame
frame = cv2.imread('frame.jpg')
frame_normalized = normalize_frame(frame)
frame_bev = transformer.transform_frame(frame_normalized)

# Detect lanes
detector = HybridLaneDetector()
results = detector.detect(frame_bev)

# Visualize each stage
print("Stage 1: Original Frame")
cv2.imshow('1_Original', frame)

print("Stage 2: Normalized Frame")
cv2.imshow('2_Normalized', frame_normalized)

print("Stage 3: Bird's Eye View")
cv2.imshow('3_BEV', frame_bev)

print("Stage 4: Edge Detection")
edge_map = results['edge_map']
if edge_map is not None:
    cv2.imshow('4_Edges', edge_map)

print("Stage 5: Detected Lines")
img_lines = frame_bev.copy()
for line in results['all_lines']:
    x1, y1, x2, y2 = line
    cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.imshow('5_Lines', img_lines)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Parameter Tuning

```python
# tune_parameters.py
import cv2
import numpy as np
from lane_detection.lane_detection import TraditionalLaneDetector

# Load test image
image = cv2.imread('test_bev_frame.jpg')

def detect_and_show(canny_low, canny_high, hough_threshold):
    detector = TraditionalLaneDetector(
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold
    )
    lines, edges = detector.detect(image)
    
    print(f"Canny: {canny_low}-{canny_high}, Hough: {hough_threshold} -> {len(lines)} lines")
    
    # Visualize
    img_lines = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow(f'Lines (H={hough_threshold})', img_lines)

# Try different parameters
for hough_thresh in [30, 50, 75, 100]:
    detect_and_show(50, 150, hough_thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Output Validation

### Validate CSV Output

```python
# validate_csv.py
import pandas as pd
import numpy as np

df = pd.read_csv('processing/lane_measurements.csv')

print("CSV Validation Report")
print("="*50)

# Check structure
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nColumn names:")
for col in df.columns:
    print(f"  - {col}")

# Check data types
print(f"\nData types:")
print(df.dtypes)

# Check for missing values
print(f"\nMissing values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found")

# Validate lane count range
print(f"\nLane count statistics (raw):")
print(df['num_lanes_raw'].describe())

# Validate lane widths
print(f"\nLane width statistics (raw, cm):")
for col in df.columns:
    if 'lane_width' in col and 'raw' in col:
        widths = pd.to_numeric(df[col], errors='coerce')
        if not widths.empty and widths.notna().sum() > 0:
            print(f"  {col}: mean={widths.mean():.2f}, std={widths.std():.2f}")

# Check confidence scores
print(f"\nConfidence statistics (raw):")
print(df['confidence_raw'].describe())

# Validate consistency across methods
print(f"\nMethod comparison (average num_lanes):")
print(f"  Raw: {df['num_lanes_raw'].mean():.2f}")
print(f"  Kalman: {df['num_lanes_kalman'].mean():.2f}")
print(f"  Temporal Avg: {df['num_lanes_temporal_avg'].mean():.2f}")
print(f"  Polynomial: {df['num_lanes_polynomial'].mean():.2f}")
```

### Compare Detection Methods

```python
# compare_methods.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('processing/lane_measurements.csv')

# Plot lane count over time per method
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, method in zip(axes.flat, ['raw', 'kalman', 'temporal_avg', 'polynomial']):
    col = f'num_lanes_{method}'
    df.plot(y=col, ax=ax, title=f'Lane Count - {method}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Number of Lanes')
    ax.grid(True)

plt.tight_layout()
plt.savefig('lane_count_comparison.png')
plt.show()

# Plot lane width distribution per method
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for ax, method in zip(axes.flat, ['raw', 'kalman', 'temporal_avg', 'polynomial']):
    widths = []
    for col in df.columns:
        if f'lane_width' in col and method in col:
            widths.extend(pd.to_numeric(df[col], errors='coerce').dropna().values)
    
    if widths:
        ax.hist(widths, bins=50, edgecolor='black')
        ax.set_title(f'Lane Width Distribution - {method}')
        ax.set_xlabel('Width (cm)')
        ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('lane_width_comparison.png')
plt.show()
```

## Performance Benchmarking

```python
# benchmark.py
import time
import cv2
from lane_detection.calibration import CalibrationManager
from lane_detection.video_processor import VideoProcessor
import config

# Initialize
start_time = time.time()

cal_manager = CalibrationManager(config.CALIBRATION_IMAGE_PATH)
H, scale = cal_manager.calibrate()
calibration_time = time.time() - start_time

print(f"Calibration time: {calibration_time:.2f}s")

# Process video
processor = VideoProcessor(cal_manager, config.__dict__)
start_time = time.time()

measurements = processor.process_video('test_video.mp4', frame_sampling=1)
processing_time = time.time() - start_time

print(f"\nProcessing Statistics:")
print(f"  Total time: {processing_time:.2f}s")
print(f"  Frames processed: {len(measurements)}")
print(f"  FPS: {len(measurements) / processing_time:.2f}")

# Get video info
from lane_detection.utils import get_video_info
video_info = get_video_info('test_video.mp4')
print(f"\nVideo Statistics:")
print(f"  Original FPS: {video_info['fps']:.2f}")
print(f"  Resolution: {video_info['width']}x{video_info['height']}")
print(f"  Total frames: {video_info['total_frames']}")
print(f"  Duration: {video_info['duration_seconds']:.2f}s")
```

## Common Issues and Solutions

### Issue: Detected lanes have high variance

**Diagnosis**:
```python
# Check variance in lane widths
import pandas as pd
import numpy as np

df = pd.read_csv('lane_measurements.csv')
for col in df.columns:
    if 'lane_width' in col and 'raw' in col:
        widths = pd.to_numeric(df[col], errors='coerce')
        cv = widths.std() / widths.mean() if widths.mean() > 0 else np.inf
        print(f"{col}: CV={cv:.2f} (should be < 0.1 for consistent lanes)")
```

**Solutions**:
1. Increase `TEMPORAL_SMOOTHING_WINDOW` in config
2. Increase `KALMAN_MEASUREMENT_NOISE` to trust smoothing more
3. Ensure good lighting conditions in video
4. Check calibration quality

### Issue: Lanes not detected in some frames

**Diagnosis**:
```python
# Check detection rate
df = pd.read_csv('lane_measurements.csv')
detection_rate = (df['num_lanes_raw'] > 0).sum() / len(df) * 100
print(f"Detection rate: {detection_rate:.1f}%")
```

**Solutions**:
1. Lower `HOUGH_THRESHOLD` to detect weaker lines
2. Lower `CANNY_LOW_THRESHOLD` for edge detection
3. Adjust `MIN_LANE_WIDTH_CM` and `MAX_LANE_WIDTH_CM`
4. Check video frame rate - may need lower `FRAME_SAMPLING`

### Issue: Phantom lanes detected

**Diagnosis**:
```python
# Check for outlier lane counts
df = pd.read_csv('lane_measurements.csv')
print(f"Lane count range: {df['num_lanes_raw'].min()} to {df['num_lanes_raw'].max()}")
print(f"Mode: {df['num_lanes_raw'].mode()[0]}")
```

**Solutions**:
1. Increase `HOUGH_THRESHOLD` to filter weak detections
2. Increase `CANNY_HIGH_THRESHOLD` for stricter edge detection
3. Filter based on `MIN_LANE_WIDTH_CM` and `MAX_LANE_WIDTH_CM`
4. Post-process CSV to remove outliers

## Logging and Tracing

Enable detailed logging:

```python
# In any module
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add file handler
fh = logging.FileHandler('lane_detection.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Now all logs will be saved to file
```

## Continuous Integration Testing

Create a simple test suite:

```python
# test_suite.py
import unittest
from lane_detection.calibration import CalibrationManager
from lane_detection.perspective_transform import PerspectiveTransformer
import numpy as np

class TestCalibration(unittest.TestCase):
    def test_calibration_loads(self):
        cal = CalibrationManager('calibration.jpg')
        self.assertIsNotNone(cal.image)
    
    def test_homography_is_valid(self):
        cal = CalibrationManager('calibration.jpg')
        H, scale = cal.calibrate()
        self.assertEqual(H.shape, (3, 3))
        self.assertGreater(scale, 0)

class TestPerspective(unittest.TestCase):
    def test_point_transformation(self):
        H = np.eye(3)
        transformer = PerspectiveTransformer(H, (800, 800), 2.0)
        points = np.array([[100, 100]])
        transformed = transformer.transform_points(points)
        np.testing.assert_array_almost_equal(points, transformed)

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest test_suite.py -v
```

---

**For more help, check README.md or enable DEBUG_MODE in config.py**
