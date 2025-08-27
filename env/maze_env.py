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
        self.consecutive_forwards = 0  # Track consecutive forward actions
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
        """Continuously grab frames and check for black, red, green, and yellow.
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
                    yellow_count = self._detect_yellow_count(frame)  # Add this line

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

                    # STRICT SEQUENTIAL LOGIC: RED → YELLOW → GREEN → RED → YELLOW → GREEN...
                    # RED → only allowed from None (start) or GREEN (completing cycle)
                    elif red_count > 150 and self.last_checkpoint in [None, "green"]:
                        self.last_checkpoint = "red"
                        print("[MazeEnv] RED checkpoint detected! +10 reward")
                        self._latest_reward += 10.0  # ← ACCUMULATES

                    elif yellow_count > 150 and self.last_checkpoint == "red":
                        self.last_checkpoint = "yellow"
                        print("[MazeEnv] YELLOW checkpoint detected! +5 reward")
                        self._latest_reward += 10.0   # ← ACCUMULATES

                    elif green_count > 150 and self.last_checkpoint == "yellow":
                        self.last_checkpoint = "green"
                        print("[MazeEnv] GREEN checkpoint detected! +10 reward")
                        self._latest_reward += 10.0  # ← ACCUMULATES

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
        self._latest_reward = 0.0
        self.consecutive_forwards = 0
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
        # DON'T execute action if motors are stopped
        if not self._motors_stopped:
            self.motor.execute_action(action)
        
        # snapshot last frame safely
        with self._frame_lock:
            frame_copy = self.last_frame.copy() if self.last_frame is not None else None
        
        processed = preprocess_frame(frame_copy) if frame_copy is not None else None
        
        if action == 0:  # 0 = forward
            self.consecutive_forwards += 1
        else:
            self.consecutive_forwards = 0
        # initialize reward and done
        done = False
        reward = 0.0
        info = {}
        
        # check camera-thread terminal (black)
        if self.stop_flag:
            done = True
            if self.last_action == 0 and self.consecutive_forwards >= 8:  # 8+ straight moves
                base_penalty = -45.0
                progressive_penalty = -(self.consecutive_forwards - 7) * 2.0  # -2 per extra forward
                reward = base_penalty + progressive_penalty
                info["reason"] = f"black_after_{self.consecutive_forwards}_forwards"
                print(f"🚫 PROGRESSIVE PENALTY: {self.consecutive_forwards} consecutive forwards = {reward} penalty!")
            elif self.last_action == 0:  # Fewer forwards but still straight
                reward = -45.0
                info["reason"] = "black_after_forward"
                print(f"🚫 STANDARD PENALTY: Hit black after {self.consecutive_forwards} forward moves = {reward}")
            else:
                reward = -40.0  # Less penalty if turning into black
                info["reason"] = "black"
                print("🚫 MINOR PENALTY: Hit black while turning")


            self.stop_flag = False
            self.motor.stop()  # Safe to call multiple times
            self._motors_stopped = True
            self.consecutive_forwards = 0
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
        upper_black = np.array([180, 50, 50])
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
        lower_green = np.array([45, 60, 60])     # Lower sat/val, start after yellow ends
        upper_green = np.array([85, 255, 255])  # Still green but more selective
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return int(cv2.countNonZero(mask))
    
    def _detect_yellow_count(self, frame):
        bottom = self.get_bottom_third(frame)
        if bottom is None:
            return 0
        try:
            hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        except Exception:
            return 0
        
        # Yellow HSV range
        lower_yellow = np.array([15, 80, 80])    # Lower sat/val thresholds, wider hue
        upper_yellow = np.array([34, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
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