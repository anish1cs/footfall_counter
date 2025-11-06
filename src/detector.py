# src/detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    """Detector class for identifying people in frames using YOLOv8."""
    
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the detector with a YOLOv8 model."""
        self.model = YOLO(model_path)
        
    def detect(self, frame):
        """
        Detect people in a frame.
        
        Args:
            frame: numpy array image in BGR format
            
        Returns:
            list of [x1, y1, x2, y2, confidence] coordinates
        """
        results = self.model.predict(frame, classes=[0], verbose=False)[0]  # class 0 is person
        boxes = []
        
        if results.boxes is not None:
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box
                if conf > 0.4:  # confidence threshold
                    boxes.append([
                        int(x1), int(y1), int(x2), int(y2), float(conf)
                    ])
                    
        return np.array(boxes) if boxes else np.array([])
