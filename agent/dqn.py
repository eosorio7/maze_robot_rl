import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import math

class ConvDQN(nn.Module):
    def __init__(self, input_shape, act_dim): #Channel height and width for input
        super(ConvDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # e.g., 3 x 64 x 64 → 32 x 15 x 15
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # → 64 x 6 x 6
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # → 64 x 4 x 4
            nn.ReLU()
        )
        print(f"[Agent] Creating dummy input for conv with shape: {input_shape}")
        self.fc_input_dim = self._get_conv_output(input_shape)
        print(f"[Agent] Got fc_input_dim: {self.fc_input_dim}")
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )

    def _get_conv_output(self, shape):
        print(f"[Agent] Entering _get_conv_output with shape: {shape}")
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, *shape, dtype=torch.float32)
                print(f"[Agent] Dummy input shape: {dummy_input.shape}")
                o = self.conv(dummy_input)
                print(f"[Agent] Output shape after conv: {o.shape}")
                return int(np.prod(o.size()))
        except Exception as e:
            print(f"[Agent] CRASHED during conv output shape calc: {e}")
            raise

    def forward(self, x):
        x = x / 255.0  # normalize image
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class DuelingDQN(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        advantage = self.advantage(x)
        # Combine value and advantage streams into Q-values
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_vals
    
class Agent:
    def __init__(self, obs_dim, act_dim, dueling = False):
        if dueling:
            self.online_net = DuelingDQN(obs_dim, act_dim)
            self.target_net = DuelingDQN(obs_dim, act_dim)
        else:
            self.online_net = ConvDQN(obs_dim, act_dim)
            self.target_net = ConvDQN(obs_dim, act_dim)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
        self.memory = []  # List of (s, a, r, s', done)
        self.gamma = 0.99
        self.batch_size = 64
        self.act_dim = act_dim
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state)
        return int(torch.argmax(q_values))

    def store(self, experience):
        self.memory.append(experience)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # FIX: Squeeze extra dimension if it exists
        if states.dim() == 5 and states.size(1) == 1:
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Double_Agent:
    def __init__(self, obs_dim, act_dim, dueling=False,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=20):
        if dueling :
            self.online_net = DuelingDQN(obs_dim, act_dim)
            self.target_net = DuelingDQN(obs_dim, act_dim)
        else:
            self.online_net = ConvDQN(obs_dim, act_dim)
            self.target_net = ConvDQN(obs_dim, act_dim)

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-3)
        self.memory = []  # List of (s, a, r, s', done)
        self.gamma = 0.99
        self.batch_size = 64
        self.act_dim = act_dim
        self.update_target()

        # Epsilon-greedy settings
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes
        self.current_episode = 0  # Will be updated externally

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


    def act(self, state):
        # Compute decayed epsilon based on current episode
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.current_episode / (self.eps_decay_episodes ))
        if random.random() < epsilon:
            return random.randint(0, self.act_dim - 1)
        state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 3:  # (C, H, W) → add batch dimension
            state = state.unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state)
        return int(torch.argmax(q_values))

    def store(self, state, action, reward, next_state, done):
        

        # Convert torch tensors -> numpy float32 arrays
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy().astype(np.float32)
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.detach().cpu().numpy().astype(np.float32)

        # If still numpy and have extra batch dim (1, C, H, W), squeeze it
        if isinstance(state, np.ndarray) and state.ndim == 4 and state.shape[0] == 1:
            state = np.squeeze(state, axis=0)
        if isinstance(next_state, np.ndarray) and next_state.ndim == 4 and next_state.shape[0] == 1:
            next_state = np.squeeze(next_state, axis=0)

        # Ensure action/reward/done are plain scalars
        action = int(action)
        reward = float(reward)
        done = bool(done)

        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def learn(self):
        return self.train_step()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # FIX: Squeeze extra dimension if it exists
        if states.dim() == 5 and states.size(1) == 1:
            states = states.squeeze(1)
            next_states = next_states.squeeze(1)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()