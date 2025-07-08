import cv2

def draw_timer(frame, box, seconds):
    x, y, w, h = box
    # رسم مستطیل دور فرد
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # نمایش زمان بالای سر فرد
    cv2.putText(frame, f"{seconds}s", (x, max(y-10, 10)),  # جلوگیری از خروج از کادر
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)