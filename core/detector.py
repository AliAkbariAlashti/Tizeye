import mediapipe as mp
import cv2
from config.settings import MIN_DETECTION_CONFIDENCE

class PersonDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONFIDENCE)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        return results.pose_landmarks is not None
