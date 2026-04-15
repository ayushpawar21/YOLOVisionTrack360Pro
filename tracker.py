"""
Tracker Module - YOLOv8-based object detection and tracking.

This module uses Ultralytics YOLOv8 for real-time object detection
with built-in tracking capabilities (BoT-SORT algorithm).

RECENT CHANGES (April 2026):
- Increased BORDER_THICKNESS to 4px for better visibility
- Added MODEL_NAME configuration at top for easy model switching (comment/uncomment)
- Simplified configuration with CONFIDENCE_THRESHOLD, TEXT_SCALE, TEXT_THICKNESS
- Color-coded bounding boxes by track ID for consistent identification
- YOLOv8 auto-downloads models on first use (~90MB to 1.3GB)

CONFIGURATION HOTSPOTS:
- Lines 12-18: MODEL_NAME selection (comment/uncomment models)
- Line 22: BORDER_THICKNESS = 4 (adjust for thicker/thinner boxes)
- Line 21: CONFIDENCE_THRESHOLD = 0.45 (adjust detection sensitivity)
"""

import cv2
import numpy as np
from ultralytics import YOLO

# =============================================================================
# MODEL SELECTION - Comment/Uncomment the model you want to use
# =============================================================================

# Fastest model (recommended for most use cases)
# MODEL_NAME = "yolov8n.pt"  # Nano model - ~90MB, fastest

# Uncomment for better accuracy (slower)
# MODEL_NAME = "yolov8s.pt"  # Small model - ~200MB, balanced
# MODEL_NAME = "yolov8m.pt"  # Medium model - ~400MB, more accurate
MODEL_NAME = "yolov8l.pt"  # Large model - ~800MB, high accuracy
# MODEL_NAME = "yolov8x.pt"  # XLarge model - ~1.3GB, best accuracy

# =============================================================================
# CONFIGURATION SETTINGS
# =============================================================================

CONFIDENCE_THRESHOLD = 0.45  # Detection confidence (0.0-1.0)
BORDER_THICKNESS = 2         # Thickness of bounding boxes (pixels)
TEXT_SCALE = 0.9             # Text size for labels
TEXT_THICKNESS = 2           # Text thickness

# =============================================================================


class VideoTracker:
    """YOLOv8-based video tracker with built-in object tracking."""

    def __init__(self, model_name=MODEL_NAME, conf_threshold=CONFIDENCE_THRESHOLD, device=None):
        """
        Initialize YOLOv8 tracker.
        
        Args:
            model_name: YOLOv8 model size (nano, small, medium, large, xlarge)
                       Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
            conf_threshold: Confidence threshold for detections (0-1)
            device: Device to run on ('cpu', 'cuda', '0', '1', etc.)
                   If None, auto-selects GPU if available, else CPU
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.device = device
        self.frame_count = 0
        
        # Store previous tracks for visualization
        self.tracked_objects = {}

    def track_frame(self, frame, iou_threshold=0.5, track_conf=0.5):
        """
        Process a single frame with YOLOv8 tracking.
        
        Args:
            frame: Input frame (numpy array)
            iou_threshold: IoU threshold for tracking
            track_conf: Confidence threshold for tracking
            
        Returns:
            Tuple of (annotated_frame, tracked_objects_dict, detections_list)
            where:
            - annotated_frame: Frame with tracking visualization
            - tracked_objects_dict: {track_id: (x, y, w, h, confidence, class_name)}
            - detections_list: [(x, y, w, h, confidence, class_name), ...]
        """
        self.frame_count += 1
        
        # Run YOLOv8 detection and tracking
        # persist=True keeps track IDs consistent across frames
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=iou_threshold,
            persist=True,
            verbose=False,
            device=self.device
        )
        
        tracked_objects = {}
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Extract tracking information
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x, y = int(x1), int(y1)
                w, h = int(x2 - x1), int(y2 - y1)
                
                # Get confidence
                confidence = float(box.conf[0].cpu().numpy())
                
                # Get class name
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # Get track ID (if tracking is enabled)
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].cpu().numpy())
                
                # Store in dictionaries
                if track_id is not None:
                    tracked_objects[track_id] = {
                        'bbox': (x, y, w, h),
                        'confidence': confidence,
                        'class_name': class_name,
                        'class_id': class_id
                    }
                
                detections.append({
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'class_name': class_name,
                    'class_id': class_id,
                    'track_id': track_id
                })
        
        self.tracked_objects = tracked_objects
        
        # Annotate frame with tracking info
        annotated_frame = self.annotate_frame(frame, tracked_objects, detections)
        
        return annotated_frame, tracked_objects, detections

    def annotate_frame(self, frame, tracked_objects, detections):
        """
        Annotate frame with tracking visualization.
        
        Args:
            frame: Input frame
            tracked_objects: Dictionary of tracked objects
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw tracked objects with track IDs
        for track_id, obj_info in tracked_objects.items():
            x, y, w, h = obj_info['bbox']
            confidence = obj_info['confidence']
            class_name = obj_info['class_name']
            
            # Get color for track ID (consistent across frames)
            color = self.get_color_for_track_id(track_id)
            
            # Draw bounding box with THICK border
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, BORDER_THICKNESS)
            
            # Draw label with track ID
            label = f"{class_name}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_THICKNESS)[0]
            
            # Background for text
            cv2.rectangle(
                annotated, 
                (x, y - 25), 
                (x + label_size[0] + 5, y), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated, 
                label, 
                (x + 5, y - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                TEXT_SCALE, 
                (255, 255, 255), 
                TEXT_THICKNESS
            )
        
        return annotated

    @staticmethod
    def get_color_for_track_id(track_id):
        """Get consistent color for track ID (BGR format)."""
        colors = [
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
        return colors[track_id % len(colors)]

    def get_model_info(self):
        """Get model information."""
        return {
            'model': str(self.model),
            'task': self.model.task,
            'device': str(self.model.device)
        }
