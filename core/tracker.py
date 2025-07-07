import time
from utils.timer_utils import draw_timer

class PersonTimerTracker:
    def __init__(self):
        self.active_person = None
        self.start_time = None

    def update_and_draw(self, frame, people_boxes):
        if len(people_boxes) > 0:
            if self.active_person is None:
                self.start_time = time.time()
                self.active_person = people_boxes[0]

            # فقط یک نفر را پیگیری می‌کنیم
            elapsed = int(time.time() - self.start_time)
            x, y, w, h = self.active_person
            draw_timer(frame, (x, y, w, h), elapsed)
        else:
            self.active_person = None
            self.start_time = None

        return frame
