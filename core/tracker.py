import time
from utils.timer_utils import draw_timer, get_day_time_string
from config.settings import LOGGING_ENABLED
import cv2
import os
import csv

class PresenceTracker:
    def __init__(self):
        self.present = False
        self.start_time = None
        self.total_time = 0

    def update(self, detected):
        current_time = time.time()

        if detected and not self.present:
            self.start_time = current_time
            self.present = True

        elif not detected and self.present:
            self.total_time += current_time - self.start_time
            self.present = False
            self.start_time = None

    def draw_overlays(self, frame):
        elapsed = 0
        if self.present:
            elapsed = time.time() - self.start_time + self.total_time
        else:
            elapsed = self.total_time

        return draw_timer(frame, elapsed)

    def save_logs(self):
        if not LOGGING_ENABLED:
            return

        os.makedirs("output", exist_ok=True)
        with open("output/logs.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([get_day_time_string(), round(self.total_time, 2)])
