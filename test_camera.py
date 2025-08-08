# test_camera.py
from picamera2 import Picamera2
import time
import cv2

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start(show_preview=False)

time.sleep(2)  # Wait for camera to stabilize

try:
    frame = picam2.capture_array()
    print("Captured frame successfully.")
    cv2.imwrite("test.jpg", frame)
except Exception as e:
    print(f"Failed to capture frame: {e}")