import numpy as np
import random

def epsilon_greedy_action(q_values, epsilon, act_dim):
    if random.random() < epsilon:
        return random.randint(0, act_dim - 1)
    else:
        return q_values.argmax().item()

def preprocess_frame(frame):
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0