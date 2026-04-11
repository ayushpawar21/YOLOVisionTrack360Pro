# Video Object Tracking System

A complete Python application for batch processing videos with real-time object tracking and visualization using OpenCV and centroid-based multi-object tracking (MOT).

## Features

- **Batch Processing**: Automatically process multiple videos from a folder
- **Object Detection & Tracking**: 
  - Motion-based detection (default, fast)
  - YOLO-based detection (optional, accurate)
  - Centroid-based tracking with unique ID assignment
- **Visual Annotations**:
  - Colored bounding boxes per object ID
  - Object IDs and confidence scores
  - Green border frame around video
  - Frame count and timestamp display
  - Live tracking count
- **Progress Tracking**: 
  - Video-level progress bar
  - Frame-level progress bar per video
- **Output Format**: MP4 with preserved FPS and resolution
- **Performance**: Optimized for smooth playback

## Project Structure

```
Video_Tracking/
├── main.py              # Entry point for batch processing
├── tracker.py           # Tracking algorithms (Centroid, YOLO)
├── utils.py             # Video I/O and visualization utilities
├── requirements.txt     # Python dependencies
├── videos/              # Input video folder
│   ├── video1.mp4
│   ├── video2.avi
│   └── ...
└── output/              # Output videos with tracking
    ├── tracked_video1.mp4
    ├── tracked_video2.mp4
    └── ...
```

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. Navigate to the project directory:
```bash
cd Video_Tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your video files in the `videos/` folder:
```bash
mkdir videos
# Copy your video files here
```

## Usage

### Basic Usage (Motion Detection - Recommended for Speed)

```bash
python main.py
```

This processes all videos in the `videos/` folder using motion-based detection, which is fast and works well for detecting moving objects.

### Advanced Usage (YOLO Detection - More Accurate)

```bash
python main.py --use-yolo
```

**Note**: YOLO mode requires downloading model weights first. See YOLO Setup section below.

### Custom Input/Output Directories

```bash
python main.py --input-dir my_videos --output-dir my_output
```

## Output

Processed videos are saved to the `output/` folder with the naming convention:
- Input: `video_name.mp4` 
- Output: `tracked_video_name.mp4`

Each output video includes:
- Original FPS and resolution preserved
- Bounding boxes with object IDs
- Color-coded boxes (consistent per object)
- Confidence scores
- Green border frame
- Frame count and elapsed time
- Number of currently tracked objects

## Performance Optimization

- **Motion Detection Mode** (default): ~30-60 FPS on modern hardware
- **YOLO Mode**: ~10-20 FPS (GPU recommended)
- Batch processing runs sequentially to manage memory

### Tips for Better Performance

1. Use motion detection mode for moving object tracking
2. Reduce video resolution before processing if needed
3. Ensure sufficient disk space in output folder
4. Run on GPU-enabled machine for YOLO mode

## Configuration

### Tracking Parameters (tracker.py)

- `maxDisappeared`: Frames before object is deregistered (default: 50)
- `distance_threshold`: Centroid matching distance (default: 50 pixels)
- `min_contour_area`: Minimum contour size to detect (default: 500 pixels)

### Detection Parameters (tracker.py - YOLO)

- `confidence_threshold`: Minimum confidence for detections (default: 0.5)
- `nms_threshold`: Non-maximum suppression threshold (default: 0.4)

## YOLO Setup (Optional)

For YOLO detection, download the model files:

1. Download YOLOv3-tiny weights (~35MB):
```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

2. Place these files in the project root directory

3. Run with YOLO:
```bash
python main.py --use-yolo
```

## Troubleshooting

### Issue: "No video files found"
- Ensure video files are in the `videos/` folder
- Check supported formats: `.mp4, .avi, .mov, .mkv, .flv, .wmv, .webm`

### Issue: Video codec errors
- Ensure FFmpeg is installed: `pip install opencv-python-headless`
- Or install FFmpeg system-wide

### Issue: Slow processing
- Use motion detection mode (default) instead of YOLO
- Reduce video resolution
- Check system CPU/GPU usage

### Issue: YOLO detection not working
- Verify YOLO weight files are downloaded and in project root
- Check file paths in tracker.py
- Try motion detection mode as fallback

## Technical Details

### Tracking Algorithm

The system uses **Centroid Tracking** with:
- Euclidean distance-based matching between frames
- Hungarian algorithm for optimal matching
- Automatic ID assignment and cleanup
- Configurable disappearance timeout

### Motion Detection Algorithm

- Grayscale frame differencing
- Gaussian blur and thresholding
- Morphological dilation for noise reduction
- Contour detection and filtering by area

## Code Architecture

### Module Overview

- **main.py**: VideoProcessor class for batch orchestration
- **tracker.py**: 
  - `CentroidTracker`: Multi-object tracking core
  - `YOLODetector`: YOLO-based detection wrapper
  - `VideoTracker`: Detection and tracking coordinator
- **utils.py**:
  - `VideoReader/Writer`: OpenCV wrappers
  - `FrameAnnotator`: Drawing and annotation utilities

### Key Classes

```python
# Main entry point
processor = VideoProcessor(input_dir="videos", output_dir="output")
processor.process_all_videos()

# Tracking
tracker = VideoTracker(use_yolo=False)
frame, tracked_objects, detections = tracker.track_frame(frame)

# Visualization
FrameAnnotator.draw_bounding_box(frame, x, y, w, h, object_id=1)
FrameAnnotator.draw_border(frame)
FrameAnnotator.add_timestamp(frame, frame_count, fps)
```

## Future Enhancements

- [ ] DeepSORT tracking for better re-identification
- [ ] Real-time video stream support
- [ ] Web UI for configuration
- [ ] Multi-GPU support
- [ ] Advanced filtering and smoothing
- [ ] Export tracking data to CSV/JSON

## Requirements

- opencv-python==4.8.1.78
- opencv-contrib-python==4.8.1.78
- numpy==1.24.3
- tqdm==4.66.1
- Pillow==10.0.0
- scipy (for centroid tracking)

## License

This project is provided as-is for educational and commercial use.

## Support

For issues or questions, check the code comments and docstrings for detailed information about each component.
