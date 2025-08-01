import time
from vision.camera_input import Camera
from control.motor_driver import MotorController
from utils import preprocess_frame
import cv2
import numpy as np

class MazeEnv:
    def __init__(self):
        self.camera = Camera()
        self.motor = MotorController()
        self.last_frame = None

    def reset(self):
        input("Place the robot at the starting line. Press Enter to continue...")
        frame = self.camera.get_frame()
        self.last_frame = frame
        processed = preprocess_frame(frame)
        return processed

    def step(self, action):
        self.motor.execute_action(action)
        time.sleep(0.2)  # Give time to move

        frame = self.camera.get_frame()
        self.last_frame = frame
        processed = preprocess_frame(frame)

        reward, done = self.compute_reward(frame)
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

    def detect_black(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 30)
        return black_pixels > 5000  # tune this

    def detect_red_finish(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return np.sum(mask) > 3000

    def detect_green_checkpoint(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([50, 100, 100])
        upper_green = np.array([70, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return np.sum(mask) > 3000