# Getting Started - Video Object Tracking

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Videos

Place your video files in the `videos/` folder:
- Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`

Example:
```bash
cp my_video.mp4 videos/
```

### 3. Run the Application

```bash
python main.py
```

Watch the progress bars as videos are processed. Output files appear in `output/` folder.

## What You Get

For each input video, you'll get:
- **Tracked video** with bounding boxes around detected objects
- **Unique IDs** for each tracked object (consistent colors)
- **Frame count and timestamp** overlay
- **Confidence scores** for detections
- **Green border** around video frame

## Next Steps

### Try Different Detection Modes

**Motion Detection (Default, Fast)**
```bash
python main.py
```
Best for: Videos with clear moving objects

**YOLO Detection (Accurate, Slower)**
First, download YOLO weights:
```bash
wget https://pjreddie.com/media/files/yolov3-tiny.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
```

Then run:
```bash
python main.py --use-yolo
```
Best for: Complex scenes with multiple object types

### Custom Directories

```bash
python main.py --input-dir my_videos --output-dir my_results
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No video files found | Check that videos are in `videos/` folder with correct extension |
| Slow processing | Use motion detection (default) instead of YOLO; reduce video resolution |
| Import errors | Run `pip install -r requirements.txt` again |
| Codec errors | Reinstall opencv: `pip install --upgrade opencv-python` |

## Understanding the Output

Example output video includes:

```
[Green Border Frame]
Frame: 125 | Time: 5.21s
Tracked Objects: 3

  +----+ ID 0 (0.95)
  | Obj|  <- Bounding box with unique ID and confidence
  +----+

  +-------+ ID 1 (0.87)
  |  Obj  |
  +-------+

  +--------+ ID 2 (0.92)
  |   Obj  |
  +--------+
```

## Configuration

Adjust tracking sensitivity in `tracker.py`:

- Line 20: `maxDisappeared = 50` - How long to remember lost objects
- Line 97: `distance_threshold = 50` - How far objects can move between frames
- Line 179: `min_contour_area = 500` - Minimum object size (motion mode)

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Entry point - runs batch processing |
| `tracker.py` | Object detection and tracking algorithms |
| `utils.py` | Video reading/writing and drawing utilities |
| `requirements.txt` | Python package dependencies |
| `README.md` | Complete documentation |

## Example Workflow

```bash
# 1. Create project folders
mkdir Video_Tracking
cd Video_Tracking

# 2. Clone/download files
# (Copy all files from project)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add videos
cp ~/Downloads/*.mp4 videos/

# 5. Run processing
python main.py

# 6. Check results
ls output/
# Output: tracked_video1.mp4, tracked_video2.mp4, ...

# 7. Try YOLO (optional)
# Download YOLO weights first
python main.py --use-yolo
```

## Performance Tips

1. **Use motion detection for speed**: Default mode processes ~30-60 FPS
2. **Run on GPU**: For YOLO mode, GPU significantly speeds up detection
3. **Reduce resolution**: Pre-process videos to lower resolution if available
4. **Close other applications**: Frees up system resources

## Next Advanced Steps

Once comfortable with basic usage:
- Modify tracking parameters for your use case
- Customize visualization (colors, annotations)
- Extend with new detection methods
- Export tracking data to CSV/JSON
- Integrate with other applications

## Need Help?

1. Check [README.md](README.md) for detailed documentation
2. Review code comments in `tracker.py`, `utils.py`, `main.py`
3. Check [.github/copilot-instructions.md](.github/copilot-instructions.md) for architecture details

---

**Happy Tracking!**
