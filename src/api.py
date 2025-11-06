# src/api.py
from flask import Flask, request, jsonify, send_file, url_for
import os
import cv2
import tempfile
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import queue
from src.detector import PersonDetector
from src.tracker import ObjectTracker
from src.counter import PersonCounter
from src.visualization import draw_overlays

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'input'
OUTPUT_FOLDER = 'output'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Job queue and results storage
processing_queue = queue.Queue()
processing_results = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path, job_id):
    """Process video and count people."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Initialize components
        detector = PersonDetector()
        tracker = ObjectTracker()
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(height * 0.5)  # Line in the middle
        counter = PersonCounter(line_y)

        # Process video
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Prepare output video
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{job_id}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            count_in, count_out = counter.update_counts(tracks)
            
            # Draw visualization
            annotated = draw_overlays(frame, tracks, line_y, count_in, count_out)
            out.write(annotated)
            
            frame_count += 1

        cap.release()
        out.release()

        # Save results
        results = {
            'status': 'completed',
            'count_in': counter.count_in,
            'count_out': counter.count_out,
            'net_occupancy': counter.count_in - counter.count_out,
            'processed_frames': frame_count,
            'total_frames': total_frames,
            'output_video': f"processed_{job_id}.avi",
            'completion_time': datetime.now().isoformat()
        }
        
        processing_results[job_id] = results
        
    except Exception as e:
        processing_results[job_id] = {
            'status': 'failed',
            'error': str(e),
            'completion_time': datetime.now().isoformat()
        }

@app.route('/api/process-video', methods=['POST'])
def process_video_endpoint():
    """Submit a video for processing."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    try:
        # Generate unique job ID and save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_id = f"{timestamp}_{secure_filename(file.filename)}"
        video_path = os.path.join(UPLOAD_FOLDER, job_id)
        file.save(video_path)

    try:
        # Process video
        cap = cv2.VideoCapture(temp_input.name)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400

        detector = PersonDetector()
        tracker = ObjectTracker()
        line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)  # Line at middle
        counter = Counter(line_y)

        # Prepare output video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_output.name, fourcc, 30.0, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            count_in, count_out = counter.update_counts(tracks)
            annotated = draw_overlays(frame, tracks, line_y, count_in, count_out)
            out.write(annotated)

        cap.release()
        out.release()

        # Return processed video
        return send_file(temp_output.name, 
                        as_attachment=True,
                        download_name='processed_video.avi',
                        mimetype='video/x-msvideo')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up temporary files
        if os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)

@app.route('/counts', methods=['GET'])
def get_counts():
    # In a real application, you'd want to persist these counts
    return jsonify({
        'count_in': Counter.count_in if hasattr(Counter, 'count_in') else 0,
        'count_out': Counter.count_out if hasattr(Counter, 'count_out') else 0,
        'total': (Counter.count_in if hasattr(Counter, 'count_in') else 0) - 
                (Counter.count_out if hasattr(Counter, 'count_out') else 0)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
