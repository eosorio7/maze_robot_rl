from env.maze_env import MazeEnv
from agent.dqn import Double_Agent

env = MazeEnv()
print("succesfully created environment env = MazeEnv()")
agent = Double_Agent(obs_dim=(3, 84, 84), act_dim=3)
print("if this message has not been reached agent was the problem")

num_episodes = 500

for episode in range(num_episodes):
    print(f"\n[train.py] Starting episode {episode}")
    state = env.reset()
    print("[train.py] Environment reset")

    total_reward = 0

    for t in range(200):  # Max steps per episode
        try:
            print(f"[train.py] Step {t}")
            action = agent.act(state)
            print(f"[train.py] Chose action: {action}")

            next_state, reward, done, _ = env.step(action)
            print(f"[train.py] Step returned reward: {reward}, done: {done}")

            agent.store(state, action, reward, next_state, done)
            print("[train.py] Stored transition")

            agent.learn()
            print("[train.py] Agent learned")

            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode} | Total reward: {total_reward}")
                input("Press Enter to start the next episode...")

        except Exception as e:
            print(f"[train.py] Exception in step {t}: {e}")
            break

    print(f"[train.py] Episode {episode} | Total reward: {total_reward}")