# src/main.py
import cv2
import argparse
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.detector import PersonDetector
from src.tracker import ObjectTracker
from src.counter import PersonCounter
from src.visualization import draw_overlays

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    # Handle input source
    if args.input.isdigit():
        cap = cv2.VideoCapture(int(args.input))
    else:
        cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: Couldn't open video source {args.input}")
        return

    # Initialize components
    detector = PersonDetector(model_path=args.model)
    tracker = ObjectTracker()
    
    # Set line position
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = args.line if args.line else int(height * 0.5)
    counter = PersonCounter(line_y)

    # Prepare output video
    if args.save:
        ensure_dir('output')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, 30.0, (width, height))

    print("Press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.resize:
            frame = cv2.resize(frame, (640, 480))

        # Process frame
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        count_in, count_out = counter.update_counts(tracks)
        annotated = draw_overlays(frame, tracks, line_y, count_in, count_out)

        # Display and save
        cv2.imshow('Footfall Counter', annotated)
        if args.save:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if args.save:
        out.release()
    cv2.destroyAllWindows()

    # Print final counts
    print(f"\nFinal Counts:")
    print(f"Entries: {counter.count_in}")
    print(f"Exits: {counter.count_out}")
    print(f"Currently Inside: {counter.count_in - counter.count_out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Footfall Counter')
    parser.add_argument('--input', type=str, default='0',
                      help='Path to video file or camera index (default: 0 for webcam)')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                      help='Path to YOLOv8 model file (default: yolov8n.pt)')
    parser.add_argument('--line', type=int,
                      help='Y-coordinate of counting line (default: middle of frame)')
    parser.add_argument('--save', action='store_true',
                      help='Save output video')
    parser.add_argument('--output', type=str, default='output/processed_video.avi',
                      help='Output video path (default: output/processed_video.avi)')
    parser.add_argument('--resize', action='store_true',
                      help='Resize frames to 640x480')
    
    args = parser.parse_args()
    main(args)
