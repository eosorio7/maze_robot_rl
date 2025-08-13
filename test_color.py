import time
import numpy as np
import cv2
from env.maze_env import MazeEnv

def crop_bottom_fourth(frame):
    h = frame.shape[0]
    return frame[int(h * 3 / 4):, :, :]

def sample_hsv_pixels(frame, num_samples=10, seed=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv.shape
    sampled_pixels = []
    
    if seed is not None:
        np.random.seed(seed)  # For reproducible sampling
    
    coords = set()
    while len(sampled_pixels) < num_samples:
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        if (x, y) not in coords:
            coords.add((x, y))
            sampled_pixels.append(hsv[y, x].tolist())
    
    return sampled_pixels

def sample_masked_hsv_pixels(frame, mask, num_samples=10, seed=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    coords = np.column_stack(np.where(mask > 0))  # all pixels where mask is nonzero
    
    if len(coords) == 0:
        print("No pixels found in mask!")
        return []
    
    if seed is not None:
        np.random.seed(seed)

    num_samples = min(num_samples, len(coords))
    chosen_indices = np.random.choice(len(coords), size=num_samples, replace=False)
    
    sampled_pixels = []
    for idx in chosen_indices:
        y, x = coords[idx]
        sampled_pixels.append(hsv[y, x].tolist())
    
    return sampled_pixels

# Recommended tuned HSV ranges
# RED (split low & high hue ranges to avoid non-reds)
RANGE_RED1 = (np.array([0, 50, 50]),   np.array([20, 255, 255]))   # lower reds
RANGE_RED2 = (np.array([160, 50, 50]), np.array([179, 255, 255]))  # upper reds

# GREEN (tighter saturation to avoid dull colors)
RANGE_GREEN = (np.array([30, 50, 50]), np.array([90, 255, 255]))
# BLACK (allow slightly brighter darks, tolerate some saturation)
RANGE_BLACK = (np.array([0, 0, 0]),    np.array([179, 255, 50]))

env = MazeEnv()
print("[ColorTest] Environment created.")

try:
    while True:
        env.motor.execute_action(0)  # Move forward

        frame = env.camera.get_frame()
        cropped = crop_bottom_fourth(frame)
        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

        # Count pixels with tuned ranges
        red_pixels = (
            cv2.countNonZero(cv2.inRange(hsv, *RANGE_RED1)) +
            cv2.countNonZero(cv2.inRange(hsv, *RANGE_RED2))
        )
        green_pixels = cv2.countNonZero(cv2.inRange(hsv, *RANGE_GREEN))
        black_pixels = cv2.countNonZero(cv2.inRange(hsv, *RANGE_BLACK))

        # Use your original detect_* methods (boolean return)
        red_detected = env.detect_red_finish(frame)
        green_detected = env.detect_green_checkpoint(frame)
        black_detected = env.detect_black(frame)

        print(f"[ColorTest] Red pixels: {red_pixels}, Detected: {red_detected}")
        if red_detected:
            red_mask = cv2.inRange(hsv, *RANGE_RED1)
            sampled_red_pixels = sample_masked_hsv_pixels(cropped, red_mask)
            print(f"[ColorTest] Sampled Black HSV pixels: {sampled_red_pixels}")

        print(f"[ColorTest] Green pixels: {green_pixels}, Detected: {green_detected}")
        if green_detected:
            green_mask = cv2.inRange(hsv, *RANGE_GREEN)
            sampled_green_pixels = sample_masked_hsv_pixels(cropped, green_mask)
            print(f"[ColorTest] Sampled Black HSV pixels: {sampled_green_pixels}")

        print(f"[ColorTest] Black pixels: {black_pixels}, Detected: {black_detected}")
        if black_detected:
            black_mask = cv2.inRange(hsv, *RANGE_BLACK)
            sampled_black_pixels = sample_masked_hsv_pixels(cropped, black_mask)
            print(f"[ColorTest] Sampled Black HSV pixels: {sampled_black_pixels}")

except KeyboardInterrupt:
    print("\n[ColorTest] Stopping test...")
    env.motor.stop()

