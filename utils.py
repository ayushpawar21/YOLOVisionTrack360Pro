"""
Utility Module - Helper functions for video I/O, drawing, and processing.

Provides functions for reading/writing videos, drawing bounding boxes,
adding visual annotations, and other helper operations.

RECENT CHANGES (April 2026):
- Added FrameAnnotator with 10-color palette for consistent object coloring
- draw_bounding_box now supports thick borders (4px) with ID and confidence
- draw_border adds green frame around video edge
- add_timestamp displays frame count and elapsed time
- add_info_text for generic overlays
- VideoReader/VideoWriter handle MP4 codec (mp4v) with frame preservation
- validate_video_file checks file accessibility before processing
- get_video_files supports .mp4, .avi, .mov, .mkv, .flv, .wmv, .webm

KEY CLASSES:
- VideoReader: Safe frame reading with metadata
- VideoWriter: MP4 output with preserved FPS/resolution
- FrameAnnotator: Static methods for drawing and annotations

VISUALIZATION CONVENTIONS:
- Colors: BGR format (OpenCV standard)
- Thickness: 4px for boxes, 2px for text
- Text position: (10, 30) is default top-left
"""

import cv2
import os
import numpy as np
from pathlib import Path


class VideoReader:
    """Read video files with frame-by-frame access."""

    def __init__(self, video_path):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self):
        """Read next frame. Returns (success, frame)."""
        return self.cap.read()

    def get_frame_count(self):
        """Get total number of frames."""
        return self.total_frames

    def get_fps(self):
        """Get frames per second."""
        return self.fps

    def get_dimensions(self):
        """Get video dimensions (width, height)."""
        return self.width, self.height

    def release(self):
        """Release video capture."""
        self.cap.release()


class VideoWriter:
    """Write processed frames to video file with codec optimization."""

    def __init__(self, output_path, fps, width, height, codec='mp4v'):
        """
        Initialize video writer.
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            width: Frame width
            height: Frame height
            codec: Video codec (default: mp4v for MP4)
        """
        self.output_path = output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use MP4V codec for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not self.writer.isOpened():
            raise ValueError(f"Cannot create video writer: {output_path}")

    def write_frame(self, frame):
        """Write a frame to the video."""
        self.writer.write(frame)

    def release(self):
        """Finalize and close the video file."""
        self.writer.release()


class FrameAnnotator:
    """Annotate frames with bounding boxes, text, and visual elements."""

    # Color palette for tracking IDs (BGR format)
    COLORS = [
        (0, 255, 0),      # Green
        (255, 0, 0),      # Blue
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Yellow
        (128, 0, 128),    # Purple
        (255, 165, 0),    # Orange
        (165, 42, 42),    # Brown
        (192, 192, 192),  # Silver
    ]

    @staticmethod
    def get_color_for_id(object_id):
        """Get consistent color for object ID."""
        return FrameAnnotator.COLORS[object_id % len(FrameAnnotator.COLORS)]

    @staticmethod
    def draw_bounding_box(frame, x, y, w, h, object_id=None, confidence=None, thickness=2):
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            x, y, w, h: Bounding box coordinates and dimensions
            object_id: Optional object ID for coloring
            confidence: Optional confidence score
            thickness: Line thickness
            
        Returns:
            Modified frame
        """
        color = FrameAnnotator.get_color_for_id(object_id) if object_id is not None else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label
        label = f"ID {object_id}" if object_id is not None else "Detection"
        if confidence is not None:
            label += f" ({confidence:.2f})"
        
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x, y - 25), (x + text_size[0] + 5, y), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    @staticmethod
    def draw_border(frame, border_color=(0, 255, 0), thickness=3):
        """
        Draw colored border around frame.
        
        Args:
            frame: Input frame
            border_color: RGB color tuple (BGR format)
            thickness: Border thickness
            
        Returns:
            Modified frame
        """
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), border_color, thickness)
        return frame

    @staticmethod
    def add_info_text(frame, text, position=(10, 30), color=(255, 255, 255), 
                     font_scale=0.7, thickness=2):
        """
        Add text annotation to frame.
        
        Args:
            frame: Input frame
            text: Text to display
            position: (x, y) position
            color: BGR color tuple
            font_scale: Font scale
            thickness: Text thickness
            
        Returns:
            Modified frame
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
        return frame

    @staticmethod
    def add_timestamp(frame, frame_count, fps):
        """
        Add timestamp to frame.
        
        Args:
            frame: Input frame
            frame_count: Current frame number
            fps: Frames per second
            
        Returns:
            Modified frame
        """
        seconds = frame_count / fps
        timestamp = f"Frame: {frame_count} | Time: {seconds:.2f}s"
        return FrameAnnotator.add_info_text(frame, timestamp, position=(10, 30))


def validate_video_file(video_path):
    """
    Validate if video file exists and is readable.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    cap.release()
    return True


def get_video_files(input_dir):
    """
    Get all video files from directory.
    
    Args:
        input_dir: Directory path
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_files = []
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    for file in os.listdir(input_dir):
        if os.path.splitext(file)[1].lower() in video_extensions:
            video_files.append(os.path.join(input_dir, file))
    
    return sorted(video_files)


def create_output_filename(input_filename):
    """
    Create output filename based on input filename.
    
    Args:
        input_filename: Original filename
        
    Returns:
        Output filename with 'tracked_' prefix and .mp4 extension
    """
    base_name = os.path.splitext(os.path.basename(input_filename))[0]
    return f"tracked_{base_name}.mp4"
