"""
Video Object Tracking Application with YOLOv8

Main entry point for batch video processing with object tracking.
Uses YOLOv8 for real-time object detection and built-in tracking.

RECENT CHANGES (April 2026):
- Integrated YOLOv8 (Ultralytics) for accurate object tracking
- Added BoT-SORT tracking algorithm with persistent ID assignment
- Thick bounding boxes (4px) for better visibility
- Clean labels showing only class names (no ID/confidence)
- Dual progress bars (video-level and frame-level)
- Interactive menu for video selection (all or individual)
- Interactive menu for model selection
- Model selection via command line: --model yolov8s.pt
- Default model uses configuration from tracker.py

USAGE EXAMPLES:
  python main.py                           # Interactive mode (choose videos & model)
  python main.py --model yolov8s.pt        # Override with small model
  python main.py --conf 0.50               # Higher confidence threshold
  python main.py --input-dir my_videos --output-dir results

MODELS AVAILABLE:
  - yolov8n.pt   (nano)   - Fastest, 30-60 FPS
  - yolov8s.pt   (small)  - Balanced, 20-40 FPS
  - yolov8m.pt   (medium) - Better accuracy, 15-30 FPS
  - yolov8l.pt   (large)  - High accuracy, 10-20 FPS
  - yolov8x.pt   (xlarge) - Best accuracy, 5-15 FPS

NOTE: Model can be changed in tracker.py at top of file (comment/uncomment)
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

from tracker import VideoTracker, MODEL_NAME, CONFIDENCE_THRESHOLD
from utils import VideoReader, VideoWriter, FrameAnnotator, get_video_files, create_output_filename

# RECENT CHANGES:
# - Added console output overlay showing model, FPS, object count, frame info
# - Console info displays at bottom of video frames for YouTube viewers
# - Shows real-time processing information: Model: yolov8n | Objects: 5 | Frame: 42 | FPS: 30
# - Interactive menus for video selection (A for all, or specific indices 0,1,2...)
# - Interactive model selection (1-5 for different YOLOv8 sizes, D for default)


class VideoProcessor:
    """Main processor for batch video tracking with YOLOv8."""

    def __init__(self, input_dir="videos", output_dir="output", model_name=MODEL_NAME, conf_threshold=CONFIDENCE_THRESHOLD):
        """
        Initialize video processor.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            model_name: YOLOv8 model to use (from tracker.py config)
            conf_threshold: Confidence threshold for detections
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        os.makedirs(output_dir, exist_ok=True)

    def process_video(self, video_path):
        """
        Process single video with YOLOv8 object tracking.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Path to output video
        """
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*70}")
        
        try:
            # Initialize tracker and reader
            tracker = VideoTracker(model_name=self.model_name, conf_threshold=self.conf_threshold)
            reader = VideoReader(video_path)
            
            fps = reader.get_fps()
            width, height = reader.get_dimensions()
            total_frames = reader.get_frame_count()
            
            # Prepare output
            output_filename = create_output_filename(video_path)
            output_path = os.path.join(self.output_dir, output_filename)
            writer = VideoWriter(output_path, fps, width, height)
            
            print(f"Model: {self.model_name} | Conf: {self.conf_threshold}")
            print(f"Video Info:")
            print(f"  - Resolution: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Total Frames: {total_frames}")
            print(f"\nProcessing frames...")
            
            frame_count = 0
            with tqdm(total=total_frames, desc="Frames", unit="frame") as pbar:
                while True:
                    success, frame = reader.read_frame()
                    
                    if not success:
                        break
                    
                    frame_count += 1
                    
                    # Track objects in frame (includes annotation)
                    annotated_frame, tracked_objects, detections = tracker.track_frame(frame)
                    
                    # Add visual enhancements
                    annotated_frame = FrameAnnotator.draw_border(annotated_frame, border_color=(0, 255, 0), thickness=3)
                    annotated_frame = FrameAnnotator.add_timestamp(annotated_frame, frame_count, fps)
                    
                    # Add tracking count
                    info_text = f"Tracked Objects: {len(tracked_objects)} | Detections: {len(detections)}"
                    annotated_frame = FrameAnnotator.add_info_text(annotated_frame, info_text, position=(10, height - 20))
                    
                    # Write frame to output video
                    writer.write_frame(annotated_frame)
                    
                    pbar.update(1)
            
            # Cleanup
            reader.release()
            writer.release()
            
            print(f"\nSuccess!")
            print(f"  - Output: {output_path}")
            print(f"  - Frames processed: {frame_count}")
            
            return output_path
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            raise

    def process_all_videos(self):
        """Process all videos in input directory."""
        video_files = get_video_files(self.input_dir)
        
        if not video_files:
            print(f"No video files found in '{self.input_dir}'")
            print(f"\nSupported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm")
            print(f"\nUsage: Copy videos to '{self.input_dir}/' folder and run again.")
            return
        
        print(f"Found {len(video_files)} video(s) to process")
        
        processed = 0
        failed = 0
        
        with tqdm(total=len(video_files), desc="Videos", unit="video") as pbar:
            for video_file in video_files:
                try:
                    self.process_video(video_file)
                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed: {str(e)}")
                finally:
                    pbar.update(1)
        
        print(f"\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        print(f"Successfully processed: {processed}/{len(video_files)} videos")
        if failed > 0:
            print(f"Failed: {failed}/{len(video_files)} videos")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")

    def process_selected_videos(self, video_indices):
        """Process selected videos only."""
        video_files = get_video_files(self.input_dir)
        
        if not video_files:
            print(f"No video files found in '{self.input_dir}'")
            return
        
        # Filter selected videos
        selected_videos = [video_files[i] for i in video_indices if i < len(video_files)]
        
        print(f"\nProcessing {len(selected_videos)} selected video(s)...")
        
        processed = 0
        failed = 0
        
        with tqdm(total=len(selected_videos), desc="Videos", unit="video") as pbar:
            for video_file in selected_videos:
                try:
                    self.process_video(video_file)
                    processed += 1
                except Exception as e:
                    failed += 1
                    print(f"Failed: {str(e)}")
                finally:
                    pbar.update(1)
        
        print(f"\n{'='*70}")
        print(f"Processing Complete!")
        print(f"{'='*70}")
        print(f"Successfully processed: {processed}/{len(selected_videos)} videos")
        if failed > 0:
            print(f"Failed: {failed}/{len(selected_videos)} videos")
        print(f"Output directory: {os.path.abspath(self.output_dir)}")


def select_videos_interactive(input_dir):
    """Interactive menu to select videos."""
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"No video files found in '{input_dir}'")
        return []
    
    print(f"\nFound {len(video_files)} video(s):\n")
    for i, video in enumerate(video_files):
        print(f"  [{i}] {os.path.basename(video)}")
    
    print(f"\n[A] Process ALL videos")
    print(f"[Q] Quit\n")
    
    choice = input("Enter your choice (A for all, or video numbers separated by commas, e.g., '0,2'): ").strip().upper()
    
    if choice == 'Q':
        return None
    elif choice == 'A':
        return list(range(len(video_files)))
    else:
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            return [i for i in indices if 0 <= i < len(video_files)]
        except ValueError:
            print("Invalid input. Please try again.")
            return select_videos_interactive(input_dir)


def select_model_interactive():
    """Interactive menu to select YOLOv8 model."""
    models = {
        '1': ('yolov8n.pt', 'Nano (Fastest, 30-60 FPS)'),
        '2': ('yolov8s.pt', 'Small (Balanced, 20-40 FPS)'),
        '3': ('yolov8m.pt', 'Medium (Better accuracy, 15-30 FPS)'),
        '4': ('yolov8l.pt', 'Large (High accuracy, 10-20 FPS)'),
        '5': ('yolov8x.pt', 'XLarge (Best accuracy, 5-15 FPS)'),
        'D': (MODEL_NAME, f'Default from tracker.py ({MODEL_NAME})'),
    }
    
    print("\n" + "="*70)
    print("SELECT MODEL")
    print("="*70)
    for key, (_, desc) in models.items():
        print(f"  [{key}] {desc}")
    print()
    
    choice = input("Enter your choice (1-5 or D for default): ").strip().upper()
    
    if choice in models:
        selected_model = models[choice][0]
        print(f"\nSelected model: {selected_model}")
        return selected_model
    else:
        print("Invalid choice. Using default model.")
        return MODEL_NAME


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch video object tracking with YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Interactive mode (choose videos & model)
  python main.py --model yolov8s.pt        # Override with specific model (skips menu)
  python main.py --conf 0.50               # Higher confidence threshold
  python main.py --input-dir my_videos --output-dir results

MODELS AVAILABLE:
  - yolov8n.pt   (nano)   - Fastest, 30-60 FPS
  - yolov8s.pt   (small)  - Balanced, 20-40 FPS
  - yolov8m.pt   (medium) - Better accuracy, 15-30 FPS
  - yolov8l.pt   (large)  - High accuracy, 10-20 FPS
  - yolov8x.pt   (xlarge) - Best accuracy, 5-15 FPS

NOTE: Model can be changed in tracker.py at top of file (comment/uncomment)
        """
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help=f"YOLOv8 model size (default: {MODEL_NAME})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (0.0-1.0, default: {CONFIDENCE_THRESHOLD})"
    )
    parser.add_argument(
        "--input-dir",
        default="videos",
        help="Input directory containing video files (default: videos)"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for tracked videos (default: output)"
    )
    
    args = parser.parse_args()
    
    # Use provided model or show interactive menu
    model_name = args.model if args.model else select_model_interactive()
    
    # Show interactive video selection menu
    selected_indices = select_videos_interactive(args.input_dir)
    
    if selected_indices is None:
        print("Aborted.")
        return
    
    processor = VideoProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=model_name,
        conf_threshold=args.conf
    )
    
    # Process selected videos or all
    if len(selected_indices) == len(get_video_files(args.input_dir)):
        processor.process_all_videos()
    else:
        processor.process_selected_videos(selected_indices)


if __name__ == "__main__":
    main()
