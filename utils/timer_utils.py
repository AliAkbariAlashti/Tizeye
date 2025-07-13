import cv2
import time
from datetime import datetime

def draw_timer(frame, elapsed_seconds):
    timer_text = f"Time: {int(elapsed_seconds)}s"
    cv2.putText(frame, timer_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)
    return frame

def get_day_time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
