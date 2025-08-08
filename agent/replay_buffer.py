import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = np.stack(states)
        next_states = np.stack(next_states)

    # Remove extra singleton dimension if it exists, e.g., shape [64,1,3,84,84] -> [64,3,84,84]
        if states.ndim == 5 and states.shape[1] == 1:
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)

        return (
            states,
            np.array(actions),
            np.array(rewards),
            next_states,
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)