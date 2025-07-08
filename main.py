from core.detector import PersonDetector
from core.tracker import PresenceTracker
import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    detector = PersonDetector()
    tracker = PresenceTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people_present = detector.detect(frame)
        tracker.update(people_present)

        annotated_frame = tracker.draw_overlays(frame)

        cv2.imshow("People Timer", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.save_logs()  # در صورت نیاز
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
