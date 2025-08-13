from env.maze_env import MazeEnv
from agent.dqn import Double_Agent
import torch
import time
import os

MODEL_PATH = "dqn_model.pth"

def save_model():
    torch.save({
        'model_state_dict': agent.online_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict()
    }, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

env = MazeEnv()
print("Successfully created environment env = MazeEnv()")
agent = Double_Agent(obs_dim=(3, 84, 84), act_dim=3)
print("If this message has not been reached agent was the problem")

# 🔹 Load model if available
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH)
    agent.online_net.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Loaded model from {MODEL_PATH}")
else:
    print("ℹ No saved model found — starting fresh.")

num_episodes = 500
action_interval = 0.2  # seconds
last_action_time = time.time()
current_action = None

try:
    for episode in range(num_episodes):
        agent.current_episode = episode
        print(f"\n[train.py] Starting episode {episode}")
        state = env.reset()
        total_reward = 0

        # Pick first action immediately
        current_action = agent.act(state)
        last_action_time = time.time()

        for t in range(126):
            now = time.time()
            if now - last_action_time >= action_interval:
                current_action = agent.act(state)
                last_action_time = now

            next_state, reward, done, info = env.step(current_action)
            agent.store(state, current_action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            # 🔹 Occasionally print rewards during episode
            if t % 3 == 0:  # every 10 steps
                print(f"  Step {t} | Reward: {reward:.3f} | Total so far: {total_reward:.3f}")

            if done:
                reason = info.get("reason", "unknown")
                print(f"🚫 Episode {episode} ended due to {reason}! Final total reward: {total_reward:.3f}")
                input("Press ENTER to continue to next episode...")
                break

        save_model()
except KeyboardInterrupt:
    print("\n🛑 Training interrupted — saving model before exit...")
    save_model()

finally:
    env.close()
    print("Environment closed.")