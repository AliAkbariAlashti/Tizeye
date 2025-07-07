import time
from utils.timer_utils import draw_timer

class PersonTimerTracker:
    def __init__(self):
        self.active_people = {}  # دیکشنری برای ذخیره افراد و زمان شروع
        self.id_counter = 0  # برای اختصاص ID به افراد

    def update_and_draw(self, frame, people_boxes):
        current_time = time.time()
        new_active_people = {}

        for box in people_boxes:
            # اختصاص ID جدید به فرد (ساده‌سازی شده)
            person_id = self.id_counter
            self.id_counter += 1

            if person_id not in self.active_people:
                self.active_people[person_id] = {'box': box, 'start_time': current_time}

            new_active_people[person_id] = self.active_people[person_id]
            elapsed = int(current_time - new_active_people[person_id]['start_time'])
            draw_timer(frame, box, elapsed)

        self.active_people = new_active_people
        return frame