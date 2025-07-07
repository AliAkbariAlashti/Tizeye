from core.detector import HumanDetector
from core.tracker import PersonTimerTracker
import cv2

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
    detector = HumanDetector()
    tracker = PersonTimerTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        people = detector.detect(frame)
        frame = tracker.update_and_draw(frame, people)

        cv2.imshow("Cafe Presence Tracker", frame)
        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()