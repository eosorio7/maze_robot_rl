import cv2
import numpy as np
from control.motor_driver import MotorController
from vision.camera_input import Camera

class MazeEnv:
    def __init__(self):
        self.motor = MotorController()
        self.camera = Camera()
        self.done = False

    def reset(self):
        self.done = False
        # Optional: reset position manually
        frame = self.camera.get_frame()
        state = self._process_frame(frame)
        return state

    def step(self, action):
        self.motor.execute_action(action)
        frame = self.camera.get_frame()
        state = self._process_frame(frame)
        reward, done = self._compute_reward(frame)
        return state, reward, done, {}

    def _process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized / 255.0

    def _compute_reward(self, frame):
        # TEMP: placeholder reward
        return 0.0, False
