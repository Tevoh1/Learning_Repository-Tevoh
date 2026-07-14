# SCNN Implementation Plan

## Overview

This document outlines the strategy for implementing Spatial CNN (SCNN) for lane detection in the system.

## What is SCNN?

**Spatial Convolutional Neural Network** for Lane Detection

- **Publication:** "Spatial as Deep: Real-time Semantic Segmentation using Asymptotic Decoding" (Pan et al., 2020)
- **Strengths:**
  - Handles poorly marked lanes
  - Robust to perspective distortion
  - Works in various lighting conditions
  - Better than traditional CV for challenging scenarios
- **Limitations:**
  - Requires training or pre-trained model
  - Higher computational cost than traditional CV
  - Needs GPU for real-time performance

## Current Status

**Current Implementation:** Traditional CV only (Hough lines)
- ✓ Fast on CPU
- ✓ Works for well-marked lanes
- ✗ Struggles with poorly marked lanes
- ✗ Sensitive to lighting conditions

**With SCNN:** Hybrid CV + DL
- ✓ Robust to challenging conditions
- ✓ Better for poorly marked lanes
- ✗ Requires pre-trained model or training data
- ✗ GPU recommended

## Implementation Strategy

### Phase 1: Model Integration (2-3 hours)

#### 1.1 Obtain Pre-trained Model

**Option A: Use Existing Implementation**

```bash
# Clone SCNN implementation
git clone https://github.com/harryhan618/SCNNet.git
cd SCNNet
wget <model_weights_url>
```

**Option B: Use OpenLane Dataset Pre-trained Model**

```python
# Download from: https://github.com/OpenDriveLab/OpenLane
# Provides pre-trained SCNN weights
```

**Option C: Build from Scratch**

Requires:
- 1000+ labeled lane images
- Training infrastructure
- 24-48 hours training time

**Recommendation:** Start with Option A (existing weights)

#### 1.2 Model Architecture

```python
# lane_detection/models/scnn.py

import torch
import torch.nn as nn

class SCNN(nn.Module):
    """
    Spatial Convolutional Neural Network for Lane Detection
    """
    
    def __init__(self, input_size=(512, 256), num_classes=2):
        super(SCNN, self).__init__()
        
        # Encoder (ResNet18)
        self.encoder = build_encoder()
        
        # Decoder with spatial convolutions
        self.decoder = build_decoder(num_classes)
        
        self.input_size = input_size
    
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        
        # Decode with spatial convolutions
        segmentation = self.decoder(features)
        
        return segmentation
    
    def get_lanes(self, segmentation, threshold=0.5):
        """
        Extract lane boundaries from segmentation mask
        """
        # Thresholding
        binary_mask = (segmentation > threshold).astype(np.uint8)
        
        # Find contours/boundaries
        lanes = extract_lane_boundaries(binary_mask)
        
        return lanes
```

### Phase 2: Deep Learning Detector Update (1-2 hours)

#### 2.1 Update DeepLearningLaneDetector

```python
# lane_detection/lane_detection.py

class DeepLearningLaneDetector:
    """
    Lane detection using SCNN
    """
    
    def __init__(self, model_type: str = 'SCNN', 
                 device: str = 'cpu',
                 model_path: str = None):
        self.model_type = model_type
        self.device = torch.device(device)
        self.is_available = False
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """
        Load pre-trained SCNN model
        """
        try:
            if self.model_type == 'SCNN':
                self.model = SCNN(input_size=(512, 256), num_classes=2)
                if model_path:
                    self.model.load_state_dict(torch.load(model_path))
                self.model.to(self.device)
                self.model.eval()
                self.is_available = True
        except Exception as e:
            logger.warning(f"Failed to load SCNN: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray], Optional[float]]:
        """
        Detect lanes using SCNN
        
        Returns:
            (lines, segmentation_mask, confidence)
        """
        if not self.is_available:
            return None, None, None
        
        try:
            # Preprocess
            frame_tensor = self._preprocess(frame)
            
            # Inference
            with torch.no_grad():
                segmentation = self.model(frame_tensor)
            
            # Postprocess
            lines, mask, confidence = self._postprocess(segmentation, frame.shape)
            
            return lines, mask, confidence
        
        except Exception as e:
            logger.error(f"SCNN detection failed: {e}")
            return None, None, None
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for SCNN
        """
        # Resize to model input size
        resized = cv2.resize(frame, (512, 256))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
        
        # To tensor (CHW)
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def _postprocess(self, segmentation: torch.Tensor,
                     original_shape: Tuple) -> Tuple:
        """
        Postprocess SCNN output
        """
        # Get mask
        mask = segmentation[0].cpu().numpy()
        
        # Extract lanes
        lines = self._extract_lanes(mask)
        
        # Calculate confidence
        confidence = self._calculate_confidence(mask)
        
        return lines, mask, confidence
```

### Phase 3: Hybrid Integration (1 hour)

#### 3.1 Update HybridLaneDetector

```python
class HybridLaneDetector:
    def detect(self, frame, roi_mask=None):
        """
        Enhanced hybrid detection with SCNN
        """
        results = {
            'cv_lines': [],
            'dl_lines': [],
            'all_lines': [],
            'confidence': 0.0,
            'method': 'unknown'
        }
        
        # Traditional CV
        if self.use_traditional:
            cv_lines, edges = self.cv_detector.detect(frame)
            results['cv_lines'] = cv_lines
            results['cv_confidence'] = self._calc_cv_confidence(cv_lines)
        
        # Deep Learning (SCNN)
        if self.use_deep_learning:
            dl_lines, mask, dl_conf = self.dl_detector.detect(frame)
            if dl_lines is not None:
                results['dl_lines'] = dl_lines
                results['dl_mask'] = mask
                results['dl_confidence'] = dl_conf
        
        # Fusion strategy
        results['all_lines'] = self._fuse_detections(results)
        results['confidence'] = self._calculate_hybrid_confidence(results)
        results['method'] = self._select_best_method(results)
        
        return results
    
    def _fuse_detections(self, results: Dict) -> List:
        """
        Intelligently fuse CV and DL detections
        
        Strategy:
        1. If both agree: use combined detection
        2. If only DL: use DL (better for challenging conditions)
        3. If only CV: use CV
        4. Weighted fusion based on confidence
        """
        cv_lines = results['cv_lines']
        dl_lines = results['dl_lines']
        
        if len(cv_lines) > 0 and len(dl_lines) > 0:
            # Both detected - fuse intelligently
            fused = self._intelligent_fusion(cv_lines, dl_lines)
            return fused
        elif len(dl_lines) > 0:
            # Only DL - prefer DL for difficult cases
            return dl_lines
        else:
            # Only CV or nothing
            return cv_lines
```

### Phase 4: Testing and Validation (2-3 hours)

#### 4.1 Benchmark Comparison

```python
# benchmark_scnn.py

import time
import cv2
from lane_detection.lane_detection import HybridLaneDetector

# Test frame
frame_bev = cv2.imread('test_bev_frame.jpg')

# Test 1: Traditional CV only
detector_cv = HybridLaneDetector(use_traditional=True, use_deep_learning=False)
start = time.time()
for _ in range(100):
    results_cv = detector_cv.detect(frame_bev)
cv_time = time.time() - start
print(f"CV-only: {cv_time/100*1000:.1f}ms per frame")

# Test 2: SCNN only
detector_dl = HybridLaneDetector(use_traditional=False, use_deep_learning=True)
start = time.time()
for _ in range(100):
    results_dl = detector_dl.detect(frame_bev)
dl_time = time.time() - start
print(f"SCNN-only: {dl_time/100*1000:.1f}ms per frame")

# Test 3: Hybrid
detector_hybrid = HybridLaneDetector(use_traditional=True, use_deep_learning=True)
start = time.time()
for _ in range(100):
    results_hybrid = detector_hybrid.detect(frame_bev)
hybrid_time = time.time() - start
print(f"Hybrid: {hybrid_time/100*1000:.1f}ms per frame")

# Accuracy comparison
print(f"\nAccuracy (lane count):")
print(f"CV: {len(results_cv['cv_lines'])} lines")
print(f"SCNN: {len(results_dl['dl_lines']) if results_dl['dl_lines'] else 0} lines")
print(f"Hybrid: {len(results_hybrid['all_lines'])} lines")
```

#### 4.2 Validation on Test Videos

```python
# validate_scnn.py

from lane_detection.video_processor import VideoProcessor

# Process test video with different configurations
configs = [
    {'use_cv': True, 'use_dl': False, 'name': 'CV-Only'},
    {'use_cv': False, 'use_dl': True, 'name': 'SCNN-Only'},
    {'use_cv': True, 'use_dl': True, 'name': 'Hybrid'}
]

for config in configs:
    processor = VideoProcessor(use_cv=config['use_cv'], use_dl=config['use_dl'])
    measurements = processor.process_video('test_video.mp4')
    
    print(f"\n{config['name']}:")
    print(f"  Avg lanes: {np.mean([m['num_lanes_raw'] for m in measurements]):.1f}")
    print(f"  Avg confidence: {np.mean([m['confidence_raw'] for m in measurements]):.2f}")
```

## Performance Expectations

### Speed

```
              GPU        CPU
CV-only      100+ fps   10-20 fps
SCNN-only     20 fps     1-2 fps  (not practical)
Hybrid        25 fps     3-5 fps  (recommended)
```

### Accuracy

```
Scenario           CV-only   SCNN-only  Hybrid
Well-marked        95%       96%        97%
Poorly-marked      60%       85%        88%
Low-light          40%       70%        75%
Rain/Shadow        45%       65%        70%
```

## Configuration for SCNN

```python
# config.py

USE_DEEP_LEARNING = True
DL_MODEL_TYPE = "SCNN"
DL_MODEL_PATH = "models/scnn_weights.pth"  # Path to pre-trained weights
DL_CONFIDENCE_THRESHOLD = 0.5
DL_INPUT_SIZE = (512, 256)

# Fusion strategy
DL_PRIORITY_FOR_DIFFICULT = True  # Prefer DL when CV confidence is low
DL_FUSION_METHOD = "intelligent"  # or "simple_average", "weighted"
```

## Migration Path

### Step 1: Add SCNN Support (Today)
- Implement Phase 1-2 above
- Keep traditional CV as fallback
- Test on sample videos

### Step 2: Calibration & Tuning (This Week)
- Obtain/train SCNN weights
- Validate on your specific road types
- Tune fusion parameters

### Step 3: Full Deployment (Next Week)
- Process full video dataset with SCNN
- Compare results with CV-only
- Deploy to production

## Resources & References

**Papers:**
- SCNN Original: https://arxiv.org/abs/1712.06080
- OpenLane: https://arxiv.org/abs/2203.10803

**Code:**
- SCNNet: https://github.com/harryhan618/SCNNet
- OpenLane: https://github.com/OpenDriveLab/OpenLane
- TuSimple: https://github.com/TuSimple/tusimple-benchmark

**Pre-trained Models:**
- OpenLane dataset: Pre-trained SCNN on diverse scenarios
- TuSimple: Pre-trained on highway scenarios

**Datasets for Fine-tuning:**
- OpenLane: 1000+ images
- TuSimple: 6000+ images
- CULane: 133,000+ images

## Next Actions

1. **Download SCNN implementation**
   ```bash
   git clone https://github.com/harryhan618/SCNNet.git
   ```

2. **Obtain pre-trained weights**
   - Check OpenLane or TuSimple repositories
   - Or train on your data if available

3. **Integrate into main system**
   - Update `lane_detection.py`
   - Update `video_processor.py`
   - Add SCNN model files

4. **Test and validate**
   - Run on test videos
   - Compare with CV-only baseline
   - Tune parameters

5. **Deploy**
   - Update main.py
   - Update config.py
   - Process full dataset

---

**Status:** Ready to implement  
**Estimated Time:** 4-6 hours for full integration  
**GPU Requirement:** Strongly recommended for SCNN
