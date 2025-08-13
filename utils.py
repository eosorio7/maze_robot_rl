import numpy as np
import random
import cv2
import numpy as np
import torch

def epsilon_greedy_action(q_values, epsilon, act_dim):
    if random.random() < epsilon:
        return random.randint(0, act_dim - 1)
    else:
        return q_values.argmax().item()

def preprocess_frame(frame):

    # Remove alpha channel if it exists (RGBA -> RGB)
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # Resize to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    # Convert from (H, W, C) to (C, H, W)
    frame = np.transpose(frame, (2, 0, 1))

    # Normalize to [0, 1] float32
    frame = frame.astype(np.float32) / 255.0

    # Return as torch tensor WITHOUT batch dimension
    frame = torch.from_numpy(frame)

    return frame