# src/visualization.py
import cv2
import numpy as np

class Visualizer:
    def __init__(self, frame_shape, max_trajectory_len=30):
        self.frame_shape = frame_shape
        self.max_trajectory_len = max_trajectory_len
        self.trajectories = {}  # {track_id: [(x, y), ...]}
        self.heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        self.color_map = {}  # {track_id: color}

    def update_trajectories(self, tracks):
        # Update trajectories
        current_tracks = set()
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            current_tracks.add(track_id)
            
            # Calculate centroid
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # Initialize or update trajectory
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
                self.color_map[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            self.trajectories[track_id].append((cx, cy))
            
            # Limit trajectory length
            if len(self.trajectories[track_id]) > self.max_trajectory_len:
                self.trajectories[track_id] = self.trajectories[track_id][-self.max_trajectory_len:]
            
            # Update heatmap
            cv2.circle(self.heatmap, (cx, cy), 20, 0.05, -1)

        # Remove old trajectories
        current_tracks = set(current_tracks)
        old_tracks = set(self.trajectories.keys()) - current_tracks
        for track_id in old_tracks:
            del self.trajectories[track_id]
            del self.color_map[track_id]

    def draw_overlays(self, frame, tracks, line_y, count_in, count_out):
        # Create copy of frame for drawing
        output = frame.copy()

        # Draw counting line
        cv2.line(output, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)

        # Update and draw trajectories
        self.update_trajectories(tracks)
        
        # Draw trajectories
        for track_id, trajectory in self.trajectories.items():
            color = self.color_map[track_id]
            points = np.array(trajectory, dtype=np.int32)
            cv2.polylines(output, [points], False, color, 2)

        # Draw bounding boxes and IDs
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            color = self.color_map[track_id]
            
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(output, f'ID:{track_id}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw counts
        cv2.putText(output, f'In: {count_in}  Out: {count_out}  Inside: {count_in - count_out}',
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Apply heatmap
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * cv2.GaussianBlur(self.heatmap, (15, 15), 0)), 
            cv2.COLORMAP_JET
        )
        
        # Fade heatmap gradually
        self.heatmap *= 0.99
        
        # Blend heatmap with output
        alpha = 0.3
        output = cv2.addWeighted(output, 1.0, heatmap_color, alpha, 0)

        return output

def draw_overlays(frame, tracks, line_y, count_in, count_out):
    """Legacy function for compatibility"""
    if not hasattr(draw_overlays, 'visualizer'):
        draw_overlays.visualizer = Visualizer(frame.shape)
    return draw_overlays.visualizer.draw_overlays(frame, tracks, line_y, count_in, count_out)
