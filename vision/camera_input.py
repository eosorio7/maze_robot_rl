import cv2

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Change index if needed

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera failed to capture frame.")
        return frame

    def release(self):
        self.cap.release()
