"""
Video Object Tracking Application with YOLOv8

Main entry point for batch video processing with object tracking.
Uses YOLOv8 for real-time object detection and built-in tracking.

Usage:
    python main.py [--model MODEL] [--conf CONF]

Options:
    --model MODEL: YOLOv8 model size (nano, small, medium, large, xlarge)
                  Default: nano (fastest)
    --conf CONF:   Confidence threshold (0.0-1.0)
                  Default: 0.45
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm

from tracker import VideoTracker
from utils import VideoReader, VideoWriter, FrameAnnotator, get_video_files, create_output_filename


class VideoProcessor:
    """Main processor for batch video tracking with YOLOv8."""

    def __init__(self, input_dir="videos", output_dir="output", model_name="yolov8n.pt", conf_threshold=0.45):
        """
        Initialize video processor.
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory for output videos
            model_name: YOLOv8 model to use
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch video object tracking with YOLOv8",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Use YOLOv8 nano (default, fastest)
  python main.py --model yolov8s.pt        # Use YOLOv8 small (more accurate)
  python main.py --model yolov8m.pt        # Use YOLOv8 medium
  python main.py --model yolov8l.pt        # Use YOLOv8 large (most accurate)
  python main.py --conf 0.50               # Higher confidence threshold
  python main.py --input-dir my_videos --output-dir results

Available Models:
  - yolov8n.pt   (nano)   - Fastest, lower accuracy
  - yolov8s.pt   (small)  - Balanced speed/accuracy
  - yolov8m.pt   (medium) - Better accuracy
  - yolov8l.pt   (large)  - High accuracy
  - yolov8x.pt   (xlarge) - Best accuracy, slowest
        """
    )
    
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model size (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Confidence threshold (0.0-1.0, default: 0.45)"
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
    
    processor = VideoProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        conf_threshold=args.conf
    )
    
    processor.process_all_videos()


if __name__ == "__main__":
    main()
