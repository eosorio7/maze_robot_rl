import time
from vision.camera import Camera
from control.motor_driver import MotorController
from utils import preprocess_frame
import cv2
import numpy as np

class MazeEnv:
    def __init__(self):
        print("About to run self.camera = camera()")
        self.camera = Camera()
        print("About to run self.motor = motorcontroller()")
        self.motor = MotorController()
        print("self.last_frame = None")
        self.last_frame = None

    def reset(self):
        print("[MazeEnv] Reset called")
    
        # Move robot to starting position if needed
        # ... your code for robot reset (if any) ...

        print("[MazeEnv] Getting frame...")
        frame = self.camera.get_frame()
        print(f"[MazeEnv] Got frame of shape: {frame.shape}")

        print("[MazeEnv] Preprocessing frame...")
        state = preprocess_frame(frame) 
        print(f"[MazeEnv] Frame preprocessed. State shape: {state.shape}")

        return state

    def step(self, action):
        self.motor.execute_action(action)
        time.sleep(0.2)  # Give time to move

        frame = self.camera.get_frame()
        self.last_frame = frame
        processed = preprocess_frame(frame)

        reward, done = self.compute_reward(frame)

        if done:
            print("[MazeEnv] Done detected — stopping motors.")
            self.motor.stop()

        return processed, reward, done, {}

    def compute_reward(self, frame):
        if self.detect_red_finish(frame):
            return 100.0, True
        elif self.detect_black(frame):
            return -50.0, True
        elif self.detect_green_checkpoint(frame):
            return 10.0, False
        else:
            return -0.1, False
    
    def get_bottom_third(self, frame):
        height = frame.shape[0]
        return frame[int(height * 2/3):, :]  # bottom third only

    def detect_black(self, frame):
        frame = self.get_bottom_third(frame)  # crop
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Count pixels below a slightly higher threshold to handle lighting
        black_pixels = np.sum(gray < 50)  # was 30

        print(f"[MazeEnv] Black pixel count: {black_pixels}")  # DEBUG

        return black_pixels > 3000  # was 5000

    def detect_red_finish(self, frame):
        frame = self.get_bottom_third(frame)  # crop
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return np.sum(mask) > 3000

    def detect_green_checkpoint(self, frame):
        frame = self.get_bottom_third(frame)  # crop
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return np.sum(mask) > 3000
    