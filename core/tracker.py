import time
from utils.timer_utils import draw_timer

class PersonTimerTracker:
    def __init__(self):
        self.active_people = {}  # دیکشنری برای ذخیره افراد و زمان شروع
        self.id_counter = 0  # برای اختصاص ID به افراد

    def update_and_draw(self, frame, people_boxes):
        current_time = time.time()
        new_active_people = {}

        # برای هر فرد شناسایی‌شده
        for box in people_boxes:
            person_id = self.id_counter
            self.id_counter += 1

            # اگر فرد جدید است، زمان شروع را ثبت کن
            if person_id not in self.active_people:
                self.active_people[person_id] = {'box': box, 'start_time': current_time}

            new_active_people[person_id] = self.active_people[person_id]
            elapsed = int(current_time - new_active_people[person_id]['start_time'])
            draw_timer(frame, box, elapsed)

        # به‌روزرسانی لیست افراد فعال
        self.active_people = new_active_people
        return frame