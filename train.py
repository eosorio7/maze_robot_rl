from env.maze_env import MazeEnv
from agent.dqn import Double_Agent

env = MazeEnv()
agent = Double_Agent(obs_dim=(1, 84, 84), act_dim=3)

num_episodes = 500

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Max steps per episode
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        agent.learn()
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode} | Total reward: {total_reward}")
