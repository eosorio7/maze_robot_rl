import time
import cv2
import numpy as np
from picamera2 import Picamera2

class Camera:
    def __init__(self):
        print("[Camera] Initializing Camera class...")
        self.picam2 = Picamera2()
        print("[Camera] Created Picamera2 instance")

        config = self.picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        print("[Camera] Created preview config")

        self.picam2.configure(config)
        print("[Camera] Configured camera")

        self.picam2.start()
        print("[Camera] Started camera")

        print("[Camera] Warming up...")
        time.sleep(2)
        print("[Camera] Warm-up complete.")

    def get_frame(self):
        try:
            frame = self.picam2.capture_array()
            # Convert RGB → BGR so OpenCV works as expected
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"[Camera] ERROR during capture: {e}")
            return None