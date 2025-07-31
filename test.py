import torch
from agent.dqn import ConvDQN
from env.maze_env import MazeEnv

env = MazeEnv()
model = ConvDQN((1, 84, 84), act_dim=4)
model.load_state_dict(torch.load("dqn_model.pth"))
model.eval()

state = env.reset()

done = False
while not done:
    state_tensor = torch.tensor(state).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        q_values = model(state_tensor)
    action = q_values.argmax().item()
    state, reward, done, _ = env.step(action)