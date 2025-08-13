import time
import threading
from vision.camera import Camera
from control.motor_driver import MotorController
from utils import preprocess_frame
import cv2
import numpy as np

class MazeEnv:
    def __init__(self, verbose=True):
        print("Initializing MazeEnv...")
        self.camera = Camera()
        self.motor = MotorController()

        # shared frame + lock
        self.last_frame = None
        self._frame_lock = threading.Lock()

        # flags & state
        self.stop_flag = False  # True if black detected (camera thread can set)
        self.last_checkpoint = None
        self.consecutive_turns = 0
        self.last_action = None
        self._latest_reward = 0.0  # updated by camera thread

        self._motors_stopped = False

        # spam control & debug
        self.suppress_black_prints = False
        self._last_black_print_ts = 0.0
        self._black_print_cooldown = 2.0
        self.verbose = verbose

        # camera thread control
        self._running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        print("[MazeEnv] Camera thread started.")

    def _camera_loop(self):
        """Continuously grab frames and check for black, red, and green.
        Updates stop_flag and checkpoint rewards in real-time.
        """
        while self._running:
            try:
                frame = self.camera.get_frame()
                with self._frame_lock:
                    self.last_frame = frame.copy() if frame is not None else None

                if frame is not None:
                    black_count = self._detect_black_count(frame)
                    red_count = self._detect_red_count(frame)
                    green_count = self._detect_green_count(frame)

                    # BLACK → stop motors, terminal
                    if black_count > 60:
                        now = time.time()
                        if not self.suppress_black_prints and now - self._last_black_print_ts > self._black_print_cooldown:
                            print("[MazeEnv] BLACK DETECTED — STOPPING MOTORS! (camera thread)")
                            self._last_black_print_ts = now
                            self.suppress_black_prints = True

                        if not self._motors_stopped:
                            self.motor.stop()
                            self._motors_stopped = True
                        self.stop_flag = True

                    # RED → mark checkpoint reward
                    if red_count > 150 and self.last_checkpoint != "red":
                        self.last_checkpoint = "red"
                        print("[MazeEnv] RED checkpoint detected! +10 reward")
                        self._latest_reward = 10.0

                    # GREEN → mark checkpoint reward
                    if green_count > 150 and self.last_checkpoint != "green":
                        self.last_checkpoint = "green"
                        print("[MazeEnv] GREEN checkpoint detected! +10 reward")
                        self._latest_reward = 10.0

                time.sleep(0.005)
            except Exception as e:
                print("[MazeEnv] Exception in camera thread:", e)
                time.sleep(0.1)

    def reset(self):
        if self.verbose:
            print("[MazeEnv] Reset called.")
        self.suppress_black_prints = False
        self.last_checkpoint = None
        self.consecutive_turns = 0
        self.last_action = None
        self.stop_flag = False
        self._motors_stopped = False

        # Wait until we have a valid frame (avoid busy-wait)
        waited = 0.0
        while True:
            with self._frame_lock:
                frame_copy = self.last_frame.copy() if self.last_frame is not None else None
            if frame_copy is not None:
                return preprocess_frame(frame_copy)
            time.sleep(0.01)   # <-- THIS MUST BE INSIDE THE LOOP
            waited += 0.01
            if waited > 5.0:
                raise RuntimeError("Timeout waiting for camera frame in reset()")

    def step(self, action):
        # execute the action
        self.motor.execute_action(action)

        # snapshot last frame safely
        with self._frame_lock:
            frame_copy = self.last_frame.copy() if self.last_frame is not None else None

        processed = preprocess_frame(frame_copy) if frame_copy is not None else None

        # initialize reward and done
        done = False
        reward = 0.0
        info = {}

        # check camera-thread terminal (black)
        if self.stop_flag:
            done = True
            reward = -20.0
            info["reason"] = "black"
            self.stop_flag = False
            if not self._motors_stopped:
                self.motor.stop()
                self._motors_stopped = True
        else:
            # reward from camera thread checkpoint detection
            reward = getattr(self, "_latest_reward", 0.0)
            self._latest_reward = 0.0
            info["reason"] = self.last_checkpoint if reward > 0 else "none"

        # spin penalty
        if self.last_action == action and action in [1, 2]:
            self.consecutive_turns += 1
        else:
            self.consecutive_turns = 0
        self.last_action = action
        if self.consecutive_turns > 3:
            reward -= 1

        # --- this part ensures motors stop and checkpoint resets when done ---
        if done:
            if not self._motors_stopped:
                self.motor.stop()
                self._motors_stopped = True
            self.last_checkpoint = None

        return processed, float(reward), done, info

    # -------------------------
    # Reward / detection helpers
    # -------------------------
    def compute_reward(self, frame):
        """Compute reward and done. Also return debug info dict with pixel counts.
           Returns: (reward:float, done:bool, debug:dict)
        """
        debug = {"black_pixels": None, "red_pixels": None, "green_pixels": None}
        step_penalty = 0.0

        # safe counts (0 if frame None)
        black_count = self._detect_black_count(frame)
        red_count = self._detect_red_count(frame)
        green_count = self._detect_green_count(frame)
        debug["black_pixels"] = black_count
        debug["red_pixels"] = red_count
        debug["green_pixels"] = green_count

        # Verbose log every time reward is evaluated (or only when verbose)
        if self.verbose:
            print(f"[compute_reward] black:{black_count} red:{red_count} green:{green_count} last_cp:{self.last_checkpoint}")

        # priorities: black terminal -> red -> green
        if black_count > 60:
            if self.verbose:
                print("[compute_reward] BLACK -> terminal")
            return -20.0 + step_penalty, True, {"reason": "black", **debug}
        # check red
        if red_count > 150:
            if self.last_checkpoint != "red":
                self.last_checkpoint = "red"
                if self.verbose:
                    print("[compute_reward] NEW red checkpoint -> +10")
                return 10.0 + step_penalty, False, {"reason": "red", **debug}
            else:
                if self.verbose:
                    print("[compute_reward] red seen but not new -> +0")
                return step_penalty, False, {"reason": "red_seen_not_new", **debug}
        # check green
        if green_count > 150:
            if self.last_checkpoint != "green":
                self.last_checkpoint = "green"
                if self.verbose:
                    print("[compute_reward] NEW green checkpoint -> +10")
                return 10.0 + step_penalty, False, {"reason": "green", **debug}
            else:
                if self.verbose:
                    print("[compute_reward] green seen but not new -> +0")
                return step_penalty, False, {"reason": "green_seen_not_new", **debug}

        return step_penalty, False, {"reason": "none", **debug}

    # low-level detection routines return raw pixel counts (safe for None frames)
    def get_bottom_third(self, frame):
        if frame is None:
            return None
        height = frame.shape[0]
        # keep your original "bottom fourth" behavior: start at 3/4 height
        return frame[int(height * 3/4):, :]

    def _detect_black_count(self, frame):
        bottom = self.get_bottom_third(frame)
        if bottom is None:
            return 0
        try:
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 32, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        return int(cv2.countNonZero(mask))

    def _detect_red_count(self, frame):
        bottom = self.get_bottom_third(frame)
        if bottom is None:
            return 0
        try:
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0
        lower_red1 = np.array([0, 100, 80])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([160, 100, 80])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        return int(cv2.countNonZero(mask))

    def _detect_green_count(self, frame):
        bottom = self.get_bottom_third(frame)
        if bottom is None:
            return 0
        try:
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0
        lower_green = np.array([32, 67, 67])
        upper_green = np.array([87, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return int(cv2.countNonZero(mask))

    # backward-compatible names (if other code calls these)
    def detect_black(self, frame):
        return self._detect_black_count(frame) > 60

    def detect_red_finish(self, frame):
        return self._detect_red_count(frame) > 150

    def detect_green_checkpoint(self, frame):
        return self._detect_green_count(frame) > 150

    def close(self):
        """Stop camera thread cleanly."""
        self._running = False
        try:
            self.camera_thread.join(timeout=2.0)
        except Exception:
            pass
        if hasattr(self.camera, "release"):
            try:
                self.camera.release()
            except Exception:
                pass