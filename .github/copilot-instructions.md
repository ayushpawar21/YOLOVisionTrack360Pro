# Video Object Tracking - AI Coding Agent Guidelines

## Project Overview

**Purpose**: Batch video processing system for object detection, tracking, and visualization using YOLOv8.

**Core Architecture**: Modular pipeline with three layers:
- **Detection**: YOLOv8 neural network-based object detection (80 classes)
- **Tracking**: Built-in BoT-SORT algorithm with unique ID assignment
- **Visualization**: Annotated output with thick bounding boxes, timestamps, and tracking info

**Key Design Pattern**: Each module (tracker, utils) is self-contained with clear responsibilities. Main.py orchestrates the pipeline.

## Essential Knowledge for Productivity

### 1. Module Dependencies & Data Flow

```
Input Videos (videos/) 
  ↓
VideoReader (utils.py) - reads frames with metadata
  ↓
VideoTracker (tracker.py) - detects and tracks objects with YOLOv8
  ↓
FrameAnnotator (utils.py) - adds visual overlays
  ↓
VideoWriter (utils.py) - encodes output MP4
  ↓
Output Videos (output/)
```

**Critical Points**:
- VideoTracker returns three values: (annotated_frame, tracked_objects_dict, detections_list)
- tracked_objects_dict maps track_id → object info (bbox, confidence, class_name)
- All coordinates are (x, y, w, h) format
- YOLOv8 handles both detection and tracking internally

### 2. YOLOv8 Tracking Algorithm Details

**YOLOv8 + BoT-SORT** (tracker.py):
- Uses Ultralytics YOLOv8 for detection (80 object classes)
- Built-in BoT-SORT tracking algorithm for consistent IDs
- persist=True keeps track IDs consistent across frames
- Auto-downloads models on first use (~90MB-1.3GB depending on model)
- GPU acceleration if available

**Model Options** (tracker.py lines 12-18):
- yolov8n.pt (nano) - Fastest, ~90MB
- yolov8s.pt (small) - Balanced, ~200MB  
- yolov8m.pt (medium) - Accurate, ~400MB
- yolov8l.pt (large) - High accuracy, ~800MB
- yolov8x.pt (xlarge) - Best accuracy, ~1.3GB

**Configuration** (tracker.py lines 21-25):
- CONFIDENCE_THRESHOLD = 0.45 (detection confidence)
- BORDER_THICKNESS = 4 (thick bounding boxes)
- TEXT_SCALE = 0.6 (label size)
- TEXT_THICKNESS = 2 (label thickness)

### 3. Video I/O Patterns

**VideoReader** (utils.py lines 11-56):
- Wrap cv2.VideoCapture for safe frame reading
- Always call .release() to free resources
- Pre-validates file before processing
- Provides: fps, width, height, total_frames

**VideoWriter** (utils.py lines 59-87):
- Uses 'mp4v' codec (default) for MP4 output
- Creates output directories automatically
- Must call .release() to finalize file
- Preserves original FPS and resolution

**Important**: Always use VideoReader/VideoWriter classes, not raw cv2 calls, for consistency.

### 4. Visualization Standards

**FrameAnnotator** (utils.py lines 90-200):
- 10-color palette for consistent object coloring (line 105-114)
- get_color_for_track_id(track_id) maps IDs to BGR colors
- draw_bounding_box: adds thick box + label + confidence
- draw_border: green frame around video edge
- add_timestamp: frame count and elapsed time
- add_info_text: generic text overlay

**Conventions**:
- All colors are BGR (OpenCV format), not RGB
- Text position (10, 30) is top-left default
- Thickness: 4 for boxes (thick), 2 for text

### 5. Main Processing Loop Pattern

See main.py VideoProcessor class:

```python
# Per-video setup
reader = VideoReader(video_path)
tracker = VideoTracker()  # Uses model from tracker.py config
writer = VideoWriter(output_path, fps, width, height)

# Frame loop with tqdm progress
with tqdm(total=total_frames) as pbar:
    while True:
        success, frame = reader.read_frame()
        if not success: break
        
        # Core processing (includes annotation)
        annotated_frame, tracked_objects, detections = tracker.track_frame(frame)
        
        # Add metadata overlays
        annotated_frame = FrameAnnotator.draw_border(annotated_frame)
        annotated_frame = FrameAnnotator.add_timestamp(annotated_frame, frame_count, fps)
        
        writer.write_frame(annotated_frame)
        pbar.update(1)

# Cleanup
reader.release()
writer.release()
```

### 6. Project-Specific Conventions

**File Organization**:
- `videos/`: input folder (auto-created, user adds files)
- `output/`: results folder (auto-created)
- `.github/`: documentation (instructions)
- Root level: main.py, tracker.py, utils.py, requirements.txt

**Model Selection** (tracker.py top):
- Comment/uncomment different MODEL_NAME lines
- Default: yolov8n.pt (fastest)
- Override with command line: `--model yolov8s.pt`

**Naming Conventions**:
- Input: any.mp4, any.avi, etc.
- Output: tracked_any.mp4 (see utils.py create_output_filename)
- Class names: PascalCase (VideoReader, FrameAnnotator)
- Functions: snake_case (read_frame, get_video_files)

**Command Line Interface**:
- Default: `python main.py` (uses model from tracker.py)
- Override model: `python main.py --model yolov8s.pt`
- Custom paths: `python main.py --input-dir X --output-dir Y`

### 7. Error Handling Patterns

**Video Validation** (utils.py lines 215-226):
```python
def validate_video_file(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(...)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(...)
    cap.release()
    return True
```

**Common Issues**:
- Missing videos/ folder → get_video_files() returns empty list
- YOLO model download fails → Check internet connection
- CUDA out of memory → Use smaller model (n/s instead of l/x)
- Frame size mismatch → VideoWriter codec incompatibility (rare)

### 8. Performance Considerations

**Model Performance**:
- yolov8n: ~30-60 FPS (fastest, lower accuracy)
- yolov8s: ~20-40 FPS (balanced)
- yolov8m: ~15-30 FPS (better accuracy)
- yolov8l: ~10-20 FPS (high accuracy)
- yolov8x: ~5-15 FPS (best accuracy, slowest)

**Optimization Strategies**:
- Use yolov8n for speed, yolov8s for balance
- GPU acceleration if available (auto-detected)
- Reduce input resolution if processing is slow
- Keep tqdm progress bars (minimal overhead, helps UX)
- Batch processing is sequential (memory-safe)

### 9. Configuration Hotspots

**Model Selection** (tracker.py lines 12-18):
- Comment/uncomment different MODEL_NAME lines
- Default: yolov8n.pt (fastest)

**Detection Settings** (tracker.py lines 21-25):
- CONFIDENCE_THRESHOLD = 0.45 (higher = fewer false positives)
- BORDER_THICKNESS = 4 (thicker boxes)
- TEXT_SCALE = 0.6 (label size)
- TEXT_THICKNESS = 2 (label thickness)

**Output** (utils.py):
- Line 77: codec = 'mp4v' (default; avoid changing without testing)
- Line 70: os.path.dirname() creates output directory

### 10. Extension Points

**Changing Models**:
1. Edit tracker.py lines 12-18
2. Comment/uncomment desired MODEL_NAME
3. Or use command line: `python main.py --model yolov8s.pt`

**Adding New Annotations**:
1. Add static method to FrameAnnotator class
2. Call from main processing loop (around line 95-102 in main.py)
3. Example: add_tracking_trail() for motion trails

**Changing Output Format**:
1. Modify VideoWriter.codec parameter (line 77 utils.py)
2. Update create_output_filename if extension changes
3. Test with common codecs: 'mp4v' (MP4), 'MJPG' (AVI), 'X264' (advanced)

## Common Tasks

**Task: Change model for better accuracy**
→ Uncomment desired MODEL_NAME in tracker.py (lines 12-18)
→ Or run: `python main.py --model yolov8s.pt`

**Task: Make borders thicker/thinner**
→ Adjust BORDER_THICKNESS in tracker.py (line 22)

**Task: Adjust detection sensitivity**
→ Increase CONFIDENCE_THRESHOLD for fewer detections
→ Decrease for more detections (but more false positives)

**Task: Debug tracking issues**
→ Check tracked_objects dict in main.py loop (line 91)
→ Verify YOLOv8 model loaded correctly
→ Check GPU memory if using large models

**Task: Process different video formats**
→ Add extensions to get_video_files() in utils.py (line 237)

---

**Last Updated**: April 2026
**Python Version**: 3.8+
**Key Dependencies**: ultralytics, opencv-python, torch, tqdm, numpy
