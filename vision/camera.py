import time
import cv2
import numpy as np
from picamera2 import Picamera2

class Camera:
    def __init__(self):
        print("[Camera] Initializing Camera class...")
        self.picam2 = Picamera2()
        print("[Camera] Created Picamera2 instance")

        config = self.picam2.create_preview_configuration(main={"size": (640, 480)})
        print("[Camera] Created preview config")

        self.picam2.configure(config)
        print("[Camera] Configured camera")

        self.picam2.start()
        print("[Camera] Started camera")

        print("[Camera] Warming up...")
        time.sleep(2)
        print("[Camera] Warm-up complete.")

    def get_frame(self):
        print("[Camera] Capturing image...")

        try:
            frame = self.picam2.capture_array()
            print(f"[Camera] Frame captured: shape = {frame.shape}")
            return frame

        except Exception as e:
            print(f"[Camera] ERROR during capture: {e}")
            return None