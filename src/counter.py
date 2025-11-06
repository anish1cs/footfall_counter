# src/counter.py
import numpy as np

class PersonCounter:
    def __init__(self, line_y):
        self.line_y = line_y
        self.count_in = 0
        self.count_out = 0
        self.track_history = {}

    def update_counts(self, tracks):
        """Update counts based on track positions relative to counting line."""
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            prev_y = self.track_history.get(track_id, cy)

            if prev_y < self.line_y and cy >= self.line_y:
                self.count_in += 1
            elif prev_y >= self.line_y and cy < self.line_y:
                self.count_out += 1

            self.track_history[track_id] = cy

        return self.count_in, self.count_out
