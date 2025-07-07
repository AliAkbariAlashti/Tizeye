from ultralytics import YOLO
import cv2
import numpy as np

class HumanDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")  # مدل نانو برای سرعت و سبک بودن

    def detect(self, frame):
        # تغییر اندازه به 416x416 برای سرعت بیشتر
        frame_resized = cv2.resize(frame, (416, 416))
        results = self.model(frame_resized, classes=[0], conf=0.5)  # فقط انسان (class 0)

        boxes = []
        for result in results:
            for box in result.boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                # مقیاس‌بندی مختصات به اندازه اصلی فریم
                ratio_x = frame.shape[1] / 416
                ratio_y = frame.shape[0] / 416
                boxes.append((int(x * ratio_x), int(y * ratio_y), int(w * ratio_x), int(h * ratio_y)))

        return boxes