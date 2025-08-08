from vision.camera import Camera
import time

if __name__ == "__main__":
    cam = Camera()
    print("[Test] Initialized camera")

    time.sleep(1)
    print("[Test] Calling get_frame()...")
    frame = cam.get_frame()
    print("[Test] Captured frame of shape:", frame.shape)