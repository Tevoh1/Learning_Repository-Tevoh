# Implementation Guide and Quickstart

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
cd lane-detection
pip install -r requirements.txt
```

### 2. Prepare Your Data

```
C:\data\
├── videos\
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── calibration.jpg
```

### 3. Configure

Edit `config.py`:

```python
INPUT_VIDEO_FOLDER = r"C:\data\videos"
CALIBRATION_IMAGE_PATH = r"C:\data\calibration.jpg"
OUTPUT_FOLDER = r"C:\data\processing"
```

### 4. Run

```bash
python main.py
```

That's it! Results will be in `C:\data\processing\`

---

## Detailed Setup Guide

### Prerequisites

- Python 3.8+
- Git
- ~2GB disk space for dependencies
- (Optional) NVIDIA GPU with CUDA 11+ for acceleration

### Step-by-Step Installation

#### 1. Clone Repository

```bash
git clone https://github.com/Tevoh1/Learning_Repository-Tevoh.git
cd Learning_Repository-Tevoh
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**If you have CUDA available:**

```bash
# For CUDA 11.8
pip install torch torchvision torch-cuda-toolkit=11.8

# For CUDA 12.1
pip install torch torchvision torch-cuda-toolkit=12.1
```

**Verify CUDA installation:**

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

#### 4. Verify Installation

```python
python -c "import cv2; import numpy as np; import pandas as pd; print('✓ All packages installed')"
```

---

## Calibration Process in Detail

### Creating Calibration Image

**Requirements:**
- 2m × 2m white sheet (or known size)
- Flat, well-lit area
- Perpendicular to camera
- Minimum 2 meters from camera

**Process:**
1. Place white sheet on road surface
2. Mount dash camera at normal driving height/angle
3. Take photo of sheet filling most of frame
4. Ensure all 4 corners are visible
5. Save as `calibration.jpg`

**Pro Tips:**
- Use midday for even lighting
- Avoid shadows on the sheet
- Keep camera at consistent angle (will be used for all videos)
- Test on multiple angles if needed

### Running Calibration

```python
from lane_detection.calibration import CalibrationManager

cal_manager = CalibrationManager('calibration.jpg', sheet_size_cm=200)

# Option 1: Fully automatic
H, scale = cal_manager.calibrate(use_auto_detect=True, manual_confirm=False)

# Option 2: Automatic with manual confirmation
H, scale = cal_manager.calibrate(use_auto_detect=True, manual_confirm=True)

# Option 3: Manual selection
H, scale = cal_manager.calibrate(use_auto_detect=False, manual_confirm=False)
```

**Manual Corner Selection:**
When prompted, click in this order:
1. Top-left corner
2. Top-right corner
3. Bottom-right corner
4. Bottom-left corner

### Verifying Calibration

```python
from lane_detection.calibration import CalibrationManager

cal_manager = CalibrationManager('calibration.jpg')
H, scale = cal_manager.calibrate()

print(f"Calibration scale: {scale:.2f} pixels/cm")
print(f"Homography matrix:\n{H}")

# Expected scale: typically 2-5 pixels/cm depending on camera distance
```

---

## Configuration Parameters Explained

### Input/Output

```python
INPUT_VIDEO_FOLDER = r"C:\data\videos"  # Videos to process
CALIBRATION_IMAGE_PATH = r"C:\data\cal.jpg"  # Calibration image
OUTPUT_FOLDER = r"C:\data\processing"  # Results folder
```

### Frame Sampling

```python
FRAME_SAMPLING = 5  # Process every 5th frame
# 1 = all frames (slowest but most data)
# 5 = every 5th frame (default, good balance)
# 30 = every 30th frame (fastest, 1 fps sampling)
```

**Effect on Processing:**
- Processing time: inverse to sampling (5x sampling = 5x faster)
- Data density: inverse to sampling
- For 30fps video: sampling=5 → ~6 fps output

### Lane Detection Parameters

**Edge Detection (Canny):**
```python
CANNY_LOW_THRESHOLD = 50    # Lower = more edges detected
CANNY_HIGH_THRESHOLD = 150  # Ratio typically 1:3
```

**Hough Line Detection:**
```python
HOUGH_THRESHOLD = 50        # Lower = more lines detected
HOUGH_MIN_LINE_LENGTH = 50  # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 20     # Max gap to connect line segments
```

**Lane Width Validation:**
```python
MIN_LANE_WIDTH_CM = 200     # Typical highway: 300-400cm
MAX_LANE_WIDTH_CM = 600     # Avoid false positives
```

### Tracking & Smoothing

**Kalman Filter:**
```python
KALMAN_PROCESS_NOISE = 0.1        # Higher = trust measurements more
KALMAN_MEASUREMENT_NOISE = 5.0    # Higher = smooth more
```

**Temporal Smoothing:**
```python
TEMPORAL_SMOOTHING_WINDOW = 5     # Frames to average (1-10)
```

**Polynomial Fitting:**
```python
POLYNOMIAL_DEGREE = 2             # Quadratic (2) or cubic (3)
```

---

## Processing Workflow

### Phase 1: Calibration
```
Calibration Image
  ↓
White Sheet Detection (auto/manual)
  ↓
Corner Detection
  ↓
Homography Computation
  ↓
Calibration Scale Calculation
  ↓
✓ Ready for video processing
```

### Phase 2: Video Processing
```
For Each Video:
  1. Load video info (FPS, resolution)
  2. For Each Sampled Frame:
     a. Read frame
     b. Normalize (CLAHE)
     c. Perspective transform to bird's eye
     d. Lane detection (CV + DL)
     e. Lane analysis (count, width)
     f. Apply tracking/smoothing
     g. Save measurements
  3. Create annotated video/frames
```

### Phase 3: Output Generation
```
All Measurements
  ↓
Pandas DataFrame
  ↓
CSV Export
  ↓
Statistics Report
  ↓
✓ lane_measurements.csv
✓ annotated_videos/
✓ annotated_frames/
```

---

## Performance Tuning

### For Speed (Offline Batch Processing)

```python
FRAME_SAMPLING = 10  # Skip more frames
OUTPUT_ANNOTATED_VIDEO = False  # Skip video encoding
OUTPUT_SAMPLE_FRAMES = False  # Skip frame saving
DL_INPUT_SIZE = (256, 128)  # Smaller DL model input
```

**Expected performance:** 30-50 fps on GPU

### For Accuracy (Detailed Analysis)

```python
FRAME_SAMPLING = 1  # Process all frames
OUTPUT_ANNOTATED_VIDEO = True  # Generate output videos
TEMPORAL_SMOOTHING_WINDOW = 10  # More smoothing
POLYNOMIAL_DEGREE = 3  # Cubic fitting
```

**Expected performance:** 5-10 fps on GPU, <1 fps on CPU

### Memory Management

```python
# For limited RAM:
# 1. Increase FRAME_SAMPLING
# 2. Reduce video resolution if possible
# 3. Set OUTPUT_ANNOTATED_VIDEO = False
# 4. Process videos one at a time
```

---

## Output Analysis

### CSV Structure

```
Frame Level Data:
video_name | frame_number | timestamp_seconds | num_lanes_raw | num_lanes_kalman | ...

Measurements (multiple per frame):
lane_width_1_cm_raw | lane_width_1_cm_kalman | lane_width_1_cm_temporal_avg | lane_width_1_cm_polynomial
lane_width_2_cm_raw | lane_width_2_cm_kalman | lane_width_2_cm_temporal_avg | lane_width_2_cm_polynomial
...

Confidence Scores:
confidence_raw | confidence_kalman | confidence_temporal_avg | confidence_polynomial
```

### Analyzing Results

```python
import pandas as pd
import numpy as np

df = pd.read_csv('lane_measurements.csv')

# Basic statistics
print("Lane Count Statistics:")
print(df['num_lanes_raw'].describe())

# Lane width per video
for video in df['video_name'].unique():
    video_data = df[df['video_name'] == video]
    print(f"\n{video}:")
    print(f"  Frames: {len(video_data)}")
    print(f"  Avg lanes: {video_data['num_lanes_raw'].mean():.1f}")

# Method comparison
print("\nMethod Comparison (Lane Count):")
for method in ['raw', 'kalman', 'temporal_avg', 'polynomial']:
    col = f'num_lanes_{method}'
    print(f"  {method}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
```

---

## Troubleshooting

### Problem: ModuleNotFoundError

```
Error: No module named 'cv2'
```

**Solution:**
```bash
pip install opencv-contrib-python --upgrade
```

### Problem: CUDA not detected

```python
import torch
print(torch.cuda.is_available())  # Returns False
```

**Solutions:**
1. Install CUDA toolkit from NVIDIA
2. Install PyTorch with CUDA support
3. Check GPU driver version
4. System will fall back to CPU automatically

### Problem: Out of Memory

```
CUDA out of memory
```

**Solutions:**
1. Reduce FRAME_SAMPLING (e.g., 10 instead of 1)
2. Reduce video resolution
3. Close other applications
4. Use CPU instead: `config.DEVICE_TYPE = 'cpu'`

### Problem: Lanes not detected

**Diagnosis:**
```python
from lane_detection.lane_detection import TraditionalLaneDetector
import cv2

frame = cv2.imread('frame.jpg')
detector = TraditionalLaneDetector(canny_low=50, canny_high=150)
lines, edges = detector.detect(frame)
print(f"Detected {len(lines)} lines")
```

**Solutions:**
1. Lower CANNY_LOW_THRESHOLD
2. Lower HOUGH_THRESHOLD
3. Increase HOUGH_MIN_LINE_LENGTH
4. Check calibration quality

---

## Next Steps

1. **Collect Data:** Capture calibration images and videos
2. **Test Calibration:** Run calibration on test images
3. **Process Pilot Video:** Test on single video
4. **Tune Parameters:** Adjust for your conditions
5. **Batch Process:** Process all videos
6. **Analyze Results:** Review CSV and statistics
7. **Iterate:** Refine parameters based on results

---

## Support Resources

- **README.md** - Project overview and features
- **API_REFERENCE.md** - Detailed API documentation
- **TESTING_AND_DEBUGGING.md** - Testing and debugging guide
- **config.py** - Configuration reference
- **main.py** - Main entry point

---

## FAQ

**Q: Can I process videos from different cameras?**
A: Yes, but create separate calibration images for each camera setup.

**Q: What's the maximum video length I can process?**
A: Depends on available RAM and disk space. Typically 1-2 hours with frame sampling.

**Q: Can I use this on mobile/embedded devices?**
A: Currently designed for desktop/server. Could be optimized for edge devices.

**Q: How accurate are the measurements?**
A: Depends on calibration quality. Typically ±2-5cm error with good calibration.

**Q: Can I process real-time video streams?**
A: Current implementation is for offline batch processing. Real-time would require architectural changes.

**Q: What video formats are supported?**
A: MP4, AVI, MOV, MKV (any format OpenCV can read).

---

For more information, see the full documentation in README.md
