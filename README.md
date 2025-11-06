# Footfall Counter using Computer Vision

This project implements a lightweight computer-vision system that counts people entering and exiting a defined area (for example a doorway) in a video or live camera stream.

## Getting Test Videos

You can use any of the following methods to get test videos:

1. Download a sample pedestrian video:
   ```bash
   # Using PowerShell (Windows)
   Invoke-WebRequest -Uri "https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi" -OutFile "input/towncentre.avi"
   ```

2. Use your webcam (pass `--input 0` when running)

3. Record your own video using a phone or camera

4. Use sample videos from public datasets:
   - [PETS Dataset](http://www.cvg.reading.ac.uk/PETS2009/a.html)
   - [MOT Challenge](https://motchallenge.net/)
   - [Oxford Town Centre](https://exposing.ai/oxford_town_centre/)

Place your test video in the `input` directory.

## Features

- Real-time person detection using YOLOv8
- Object tracking using a SORT-like tracker (Kalman filter + data association)
- Entry/exit counting using a configurable virtual line
- Visualization overlays and optional output video file

## Technical Approach

### Person Detection
- Utilizes YOLOv8 (You Only Look Once) for real-time person detection
- Filters detections to focus only on person class with confidence > 0.4

### Object Tracking
- Implements SORT (Simple Online and Realtime Tracking) algorithm
- Maintains object IDs across frames
- Handles multiple simultaneous tracks

### Counting Logic
- Uses a virtual line crossing detection method
- Tracks the centroid of each person
- Determines direction based on line crossing events
- Maintains count of entries and exits separately

## Installation

1. Clone the repository and open the project directory:

```powershell
git clone https://github.com/yourusername/footfall-counter.git
cd footfall-counter
```

2. Create and activate a Python virtual environment (Windows PowerShell shown):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install required dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Usage

## Usage

Run the main script to process a video file or a webcam stream. Example usages (PowerShell):

Run a video file and save annotated output:

```powershell
.\venv\Scripts\python.exe src\main.py --input input\test_video.mp4 --save --output output\processed_test_video.avi --resize
```

Run the webcam (camera index 0):

```powershell
.\venv\Scripts\python.exe src\main.py --input 0 --resize
```

Common flags:
- `--input`: path to video file or camera index (default `0`)
- `--model`: path to YOLOv8 model (default `yolov8n.pt`)
- `--line`: Y coordinate for counting line (default: middle of frame)
- `--save`: save annotated output to `--output` path
- `--resize`: resize frames to 640x480 (useful for faster processing)

## Project Structure

```
footfall_counter/
├── input/                 # place test videos here (example: input/test_video.mp4)
├── output/                # processed outputs are saved here
├── src/
│   ├── main.py            # Main application entrypoint (run this)
│   ├── detector.py        # Person detection (YOLOv8)
│   ├── tracker.py         # SORT-like tracker implementation
│   ├── counter.py         # Counting logic
│   └── visualization.py   # Drawing overlays and annotations
├── requirements.txt       # Python dependencies
└── README.md
```

## Dependencies

Key dependencies (installed via `requirements.txt`):

- ultralytics (YOLOv8)
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- Matplotlib
- Pillow
- filterpy (for Kalman filter used by tracker)

The project assumes you have a YOLOv8 model file available (default `yolov8n.pt`). The `ultralytics` package will attempt to download or use bundled models if not provided.

## Performance Considerations

- With a GPU and an optimized YOLOv8 model, detection can be real-time. On CPU, performance will be slower and depends on model size and resolution.
- Use `--resize` to reduce frame size (e.g., 640x480) for faster CPU processing.
- For production or high-volume processing, consider batching or running on a machine with a CUDA-capable GPU.

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

test video url = https://youtu.be/YzcawvDGe4Y?si=kKWak_9OyRBW1A-f
