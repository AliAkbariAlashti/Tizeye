import cv2

class HumanDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        # resize for better speed
        frame_resized = cv2.resize(frame, (640, 480))
        boxes, _ = self.hog.detectMultiScale(frame_resized,
                                             winStride=(8,8),
                                             padding=(8,8),
                                             scale=1.05)
        # scale back coordinates
        ratio_x = frame.shape[1] / 640
        ratio_y = frame.shape[0] / 480
        scaled_boxes = [(int(x*ratio_x), int(y*ratio_y), int(w*ratio_x), int(h*ratio_y)) for (x,y,w,h) in boxes]
        return scaled_boxes
