# User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Measurement and Calibration](#measurement-and-calibration)
5. [Video Processing](#video-processing)
6. [Web Interfaces](#web-interfaces)
7. [Python API](#python-api)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool.git
cd Vision-Based-3D-Measurement-Perception-Tool

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### First Run

Process your first image:

```bash
python scripts/run_single_image.py path/to/image.jpg -o output/
```

This will:
- Download the MiDaS model (first time only)
- Process the image
- Save depth map, overlay, and metadata to `output/`

## Basic Usage

### Single Image Processing

**Command Line:**
```bash
# Basic processing
python scripts/run_single_image.py image.jpg

# Specify output directory
python scripts/run_single_image.py image.jpg -o my_output/

# Choose model (faster vs better quality)
python scripts/run_single_image.py image.jpg -m MiDaS_small  # Fast
python scripts/run_single_image.py image.jpg -m DPT_Hybrid   # Balanced
python scripts/run_single_image.py image.jpg -m DPT_Large    # Best quality

# Change colormap
python scripts/run_single_image.py image.jpg -c plasma
python scripts/run_single_image.py image.jpg -c jet
```

**Python API:**
```python
from src.core.depth_estimator import DepthEstimator
from src.preprocessing.image_loader import ImageLoader

# Initialize
loader = ImageLoader()
estimator = DepthEstimator(model_type="DPT_Large")

# Load and process
image = loader.load_single("image.jpg")
depth = estimator.estimate_depth(image, normalize=True)

# depth is a numpy array with shape (H, W) and values in [0, 1]
```

### Batch Processing

Process multiple images efficiently:

```bash
# Process all images in directory
python scripts/run_batch.py input_dir/ -o output_dir/

# Process only PNG files
python scripts/run_batch.py input_dir/ -p "*.png" -o output_dir/

# Use faster model for large batches
python scripts/run_batch.py input_dir/ -m MiDaS_small -o output_dir/
```

**Python API:**
```python
from src.preprocessing.image_loader import ImageLoader
from src.core.depth_estimator import DepthEstimator

loader = ImageLoader()
estimator = DepthEstimator(model_type="DPT_Hybrid", optimize=True)

# Load batch
images = loader.load_batch("input_dir/", pattern="*.jpg")

# Process batch
depths = estimator.estimate_depth_batch(images)

# depths is a list of numpy arrays
```

## Advanced Features

### Depth Map Customization

**Colormaps:**

Available colormaps:
- `viridis` - Blue to yellow (default)
- `plasma` - Purple to yellow
- `inferno` - Black to yellow
- `magma` - Black to white
- `jet` - Blue to red (classic)
- `turbo` - Rainbow
- `coolwarm` - Blue to red diverging

```python
from src.postprocessing.colormap_generator import ColormapGenerator

colormap_gen = ColormapGenerator(default_colormap='plasma')
depth_colored = colormap_gen.apply_colormap(depth, colormap='viridis')
```

**Depth Processing:**

```python
from src.postprocessing.depth_processor import DepthProcessor

processor = DepthProcessor()

# Remove outliers
depth_clean = processor.remove_outliers(depth, method='clip', percentile=95)

# Apply smoothing
depth_smooth = processor.apply_bilateral_filter(
    depth, d=9, sigma_color=75, sigma_space=75
)

# Apply Gaussian blur
depth_blur = processor.apply_gaussian_filter(depth, kernel_size=5)
```

**Overlays:**

```python
from src.visualization.overlay_renderer import OverlayRenderer

renderer = OverlayRenderer()

# Blend depth with image
overlay = renderer.blend_depth_with_image(
    image, depth, alpha=0.5, colormap='viridis'
)

# Create side-by-side comparison
comparison = renderer.create_side_by_side(
    image, depth, colormap='plasma'
)
```

## Measurement and Calibration

### Point-to-Point Distance

**Command Line:**
```bash
python scripts/measurement_demo.py image.jpg -o measurements/
```

**Python API:**
```python
from src.measurement.distance_calculator import DistanceCalculator

calc = DistanceCalculator()

# Measure distance between two points
p1 = (100, 150)  # (x, y)
p2 = (300, 250)

distance = calc.point_to_point_distance(
    depth, p1, p2, method="euclidean"
)
print(f"Distance: {distance:.2f} pixels")

# Manhattan distance
distance_manhattan = calc.point_to_point_distance(
    depth, p1, p2, method="manhattan"
)
```

### Scale Calibration

Convert relative depth to real-world measurements:

```python
from src.measurement.scale_estimator import ScaleEstimator

scale_est = ScaleEstimator()

# Use a known reference (e.g., 1 meter ruler)
known_distance = 1.0  # meters
ref_p1 = (50, 100)
ref_p2 = (250, 100)

# Calibrate
calibration = scale_est.calibrate(
    depth,
    known_distance=known_distance,
    reference_points=[ref_p1, ref_p2],
    unit="meters"
)

# Now measure other distances in meters
distance_pixels = calc.point_to_point_distance(depth, p1, p2)
distance_meters = scale_est.depth_to_meters(distance_pixels, calibration)

print(f"Distance: {distance_meters:.2f} meters")

# Save calibration for later use
scale_est.save_calibration(calibration, "my_calibration.json")

# Load calibration
calibration = scale_est.load_calibration("my_calibration.json")
```

### Region Analysis

Analyze depth in specific regions:

```python
# Define region as (x1, y1, x2, y2)
region = (100, 100, 300, 250)

analysis = calc.analyze_region_depth(depth, region)

print(f"Mean depth: {analysis['mean_depth']:.3f}")
print(f"Min depth: {analysis['min_depth']:.3f}")
print(f"Max depth: {analysis['max_depth']:.3f}")
print(f"Area: {analysis['area_pixels']} pixels")
```

### Object Detection

Detect objects based on depth:

```python
from src.measurement.object_analyzer import ObjectAnalyzer

analyzer = ObjectAnalyzer()

# Detect objects at different depths
objects = analyzer.detect_objects(
    depth,
    depth_threshold=0.5,  # Depth threshold
    min_area=500         # Minimum area in pixels
)

print(f"Found {len(objects)} objects")

for i, obj in enumerate(objects):
    print(f"Object {i+1}:")
    print(f"  Area: {obj['area']} pixels")
    print(f"  Centroid: {obj['centroid']}")
    print(f"  Mean depth: {obj['mean_depth']:.3f}")
    print(f"  Bounding box: {obj['bbox']}")
```

## Video Processing

### Video File Processing

```bash
# Process video file
python scripts/run_video.py video.mp4 -o output/

# Sample every 5th frame
python scripts/run_video.py video.mp4 -s 5 -o output/

# Save individual frames
python scripts/run_video.py video.mp4 --save-frames -o output/

# Limit number of frames
python scripts/run_video.py video.mp4 --max-frames 100 -o output/
```

**Python API:**
```python
from src.preprocessing.video_processor import VideoProcessor
from src.core.depth_estimator import DepthEstimator

# Open video
video = VideoProcessor("video.mp4")
estimator = DepthEstimator(model_type="MiDaS_small")  # Use fast model

# Process frames
for frame_num, frame in video.extract_frames(sample_rate=5):
    depth = estimator.estimate_depth(frame)
    # Process depth...

video.close()
```

### Real-Time Webcam

```bash
# Start webcam depth estimation
python scripts/run_webcam.py --camera 0 -m MiDaS_small

# Save recording
python scripts/run_webcam.py --save-recording output.mp4

# Custom resolution
python scripts/run_webcam.py --width 1280 --height 720
```

**Keyboard Controls:**
- `q`: Quit
- `s`: Save screenshot

**Python API:**
```python
from src.preprocessing.webcam_handler import WebcamHandler
from src.core.depth_estimator import DepthEstimator

# Initialize
webcam = WebcamHandler(camera_id=0, width=640, height=480)
estimator = DepthEstimator(model_type="MiDaS_small")

# Start capture
webcam.start_capture()

try:
    while True:
        frame = webcam.get_latest_frame()
        if frame is not None:
            depth = estimator.estimate_depth(frame)
            # Display or process depth...
finally:
    webcam.stop_capture()
```

## Web Interfaces

### Gradio Demo

Quick interactive demo:

```bash
# Install UI dependencies
pip install -r requirements-ui.txt

# Launch Gradio app
python app/gradio_app.py
```

Access at: http://localhost:7860

**Features:**
- Drag-and-drop image upload
- Model selection
- Click-to-measure distances
- 3D visualization
- Real-time parameter adjustments

### Streamlit Dashboard

Full-featured application:

```bash
# Launch Streamlit dashboard
streamlit run app/streamlit_app.py
```

Access at: http://localhost:8501

**Pages:**
1. **Single Image:** Upload and process images
2. **Measurements:** Distance, calibration, objects
3. **3D Visualization:** Interactive point clouds
4. **Batch Processing:** CLI command reference
5. **About:** Documentation and info

## Python API

### Complete Example

```python
# Complete workflow
from src.core.depth_estimator import DepthEstimator
from src.preprocessing.image_loader import ImageLoader
from src.postprocessing.colormap_generator import ColormapGenerator
from src.visualization.overlay_renderer import OverlayRenderer
from src.measurement.distance_calculator import DistanceCalculator
from src.measurement.scale_estimator import ScaleEstimator
from src.export.image_saver import ImageSaver
from src.export.json_exporter import JSONExporter

# 1. Load image
loader = ImageLoader()
image = loader.load_single("image.jpg")

# 2. Estimate depth
estimator = DepthEstimator(model_type="DPT_Large", optimize=True)
depth = estimator.estimate_depth(image, normalize=True)

# 3. Visualize
colormap_gen = ColormapGenerator()
depth_colored = colormap_gen.apply_colormap(depth, colormap='viridis')

overlay_renderer = OverlayRenderer()
overlay = overlay_renderer.blend_depth_with_image(image, depth, alpha=0.5)

# 4. Measure
calc = DistanceCalculator()
p1, p2 = (100, 100), (300, 200)
distance = calc.point_to_point_distance(depth, p1, p2)

# 5. Calibrate
scale_est = ScaleEstimator()
calibration = scale_est.calibrate(
    depth, known_distance=2.0, reference_points=[(50, 50), (200, 200)]
)
distance_meters = scale_est.depth_to_meters(distance, calibration)

# 6. Export
saver = ImageSaver(output_dir="output/")
paths = saver.save_depth_results(
    image=image,
    depth=depth,
    depth_colored=depth_colored,
    overlay=overlay,
    output_name="result"
)

json_exporter = JSONExporter()
metadata = json_exporter.create_metadata(
    depth=depth, image_path="image.jpg", model_type="DPT_Large"
)
json_exporter.save_metadata(metadata, "output/metadata.json")

print(f"Distance: {distance_meters:.2f} meters")
print(f"Saved to: {paths['depth_map']}")
```

## Best Practices

### Model Selection

Choose based on your use case:

| Model | Speed | Quality | RAM | Best For |
|-------|-------|---------|-----|----------|
| MiDaS_small | Fast | Good | ~2GB | Real-time, webcam |
| DPT_Hybrid | Medium | Better | ~3GB | General use |
| DPT_Large | Slow | Best | ~4GB | High-quality outputs |

### GPU Utilization

**Enable optimization:**
```python
estimator = DepthEstimator(model_type="DPT_Large", optimize=True)
```

**Check device:**
```python
print(f"Using device: {estimator.device_manager.device_name}")
```

**Batch for throughput:**
```python
# Process in batches for better GPU utilization
depths = estimator.estimate_depth_batch(images, batch_size=16)
```

### Memory Management

**Clear cache periodically:**
```python
estimator.device_manager.clear_cache()
```

**Process large images in batches:**
```python
# For very large images, resize first
from PIL import Image
img = Image.open("huge_image.jpg")
img_resized = img.resize((1920, 1080))
```

### Calibration Tips

1. **Use clear reference objects:** Rulers, standard objects
2. **Measure in center region:** More accurate than edges
3. **Multiple calibrations:** Average for better accuracy
4. **Save calibrations:** Reuse for similar conditions
5. **Re-calibrate:** When changing environments

### Measurement Accuracy

**Factors affecting accuracy:**
- Image quality (higher is better)
- Model choice (DPT_Large > DPT_Hybrid > MiDaS_small)
- Distance from camera (closer is more accurate)
- Texture and lighting (more texture = better)
- Calibration quality

**Tips:**
- Use highest quality images
- Ensure good lighting
- Avoid textureless surfaces
- Calibrate for each scene
- Multiple measurements for validation

## Troubleshooting

### Common Issues

**1. Model Download Fails**

```
Error: Failed to download model from torch.hub
```

**Solution:**
- Check internet connection
- Clear torch hub cache: `rm -rf ~/.cache/torch/hub`
- Manually download and place in cache directory

**2. Out of Memory (GPU)**

```
RuntimeError: CUDA out of memory
```

**Solution:**
- Use smaller model: `MiDaS_small`
- Reduce batch size
- Reduce image resolution
- Clear cache: `estimator.device_manager.clear_cache()`

**3. Slow Performance (CPU)**

```
Processing takes several seconds per image
```

**Solution:**
- Enable GPU if available
- Use MiDaS_small model
- Reduce image resolution
- Process in batches

**4. Depth Map Quality Issues**

**Solution:**
- Use DPT_Large for best quality
- Ensure good lighting
- Use high-resolution images
- Try different preprocessing options

**5. Import Errors**

```
ModuleNotFoundError: No module named 'src'
```

**Solution:**
- Install package: `pip install -e .`
- Check virtual environment is activated
- Verify Python path

### Debug Mode

Enable detailed logging:

```python
from src.utils.logger import setup_logger

logger = setup_logger("my_app", level="DEBUG")
```

## FAQ

**Q: Which model should I use?**

A: Start with `MiDaS_small` for speed, upgrade to `DPT_Large` for quality.

**Q: Can I use this without a GPU?**

A: Yes, but it will be slower. Use `MiDaS_small` for best CPU performance.

**Q: How accurate are the measurements?**

A: Depends on calibration and conditions. Typically ±5-10% with good calibration.

**Q: Can I process videos in real-time?**

A: Yes, use `MiDaS_small` with GPU for 30+ FPS on 640x480.

**Q: Does it work with grayscale images?**

A: Yes, grayscale images are automatically converted to RGB.

**Q: Can I fine-tune the models?**

A: Models are pretrained. Custom training requires additional work.

**Q: How do I export 3D point clouds?**

A: Use `DepthViewer.save_point_cloud()` with format='ply'.

**Q: Can I use this commercially?**

A: Yes, under MIT license. Check MiDaS license for model usage.

**Q: Where are models cached?**

A: Default: `~/.cache/torch/hub`. Configure with `model.cache_dir` in YAML.

**Q: How do I contribute?**

A: Open issues or PRs on GitHub. See CONTRIBUTING.md.

## Additional Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
<!-- - [Example Notebooks](../notebooks/) -->
- [GitHub Issues](https://github.com/yourusername/Vision-Based-3D-Measurement-Perception-Tool/issues)
- [MiDaS Repository](https://github.com/isl-org/MiDaS)

## Support

For questions or issues:
1. Check this guide and FAQ
2. Search GitHub issues
3. Open a new issue with:
   - Python version
   - OS and hardware
   - Error messages
   - Minimal reproduction code
