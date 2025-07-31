import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

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
        self.fc_input_dim = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim)
        )

    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

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

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (~dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Double_Agent:
    def __init__(self, obs_dim, act_dim, dueling = False):
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

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states)
            selected_q_values = next_q_target.gather(1, best_actions).squeeze(1)
            target_q = rewards + self.gamma * selected_q_values * (~dones)

        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()