# MAZE-ROBOT-RL

> A Raspberry-Pi-powered maze robot that learns to navigate a colored-checkpoint track using a convolutional Double-DQN.

---

## Overview

This repository contains code for a maze robot powered by a camera, motor drivers, and a Double DQN agent (PyTorch). The robot learns to pass sequential colored checkpoints (RED → YELLOW → GREEN) while avoiding large black regions (terminal condition with penalties). The project includes:

- Camera interface (`vision/camera.py`)
- Motor control (`control/motor_driver.py`)
- Environment wrapper that converts camera frames to observations and computes rewards (`env/maze_env.py`)
- Agent implementations (`agent/dqn.py`)
- Training loop (`train.py`) and evaluation script (`play.py`)
- Utilities for preprocessing frames (`utils.py`)

---

## Quick start

1. Ensure you are on a Raspberry Pi with camera support and GPIO access.
2. Install dependencies (example):
```bash
pip install torch torchvision numpy opencv-python picamera2 gpiozero
```
> Note: Installing PyTorch on a Raspberry Pi may require a prebuilt wheel for your Pi/OS or building from source.

3. From the repository root:
```bash
# Train the agent
python train.py

# Run a trained model (inference)
python play.py
```

`train.py` saves/loads the model from `dqn_model.pth` by default.

---

## Repo layout

```
.
├─ train.py                # training loop
├─ play.py                 # inference / evaluate script
├─ utils.py                # preprocess_frame, epsilon_greedy_action
├─ vision/
│  └─ camera.py            # Picamera2 wrapper
├─ env/
│  └─ maze_env.py          # MazeEnv: threading camera loop, rewards, step/reset
├─ control/
│  └─ motor_driver.py      # MotorController using gpiozero
├─ agent/
│  └─ dqn.py               # ConvDQN, Double_Agent and helpers
└─ dqn_model.pth           # saved model file used by scripts
```

---

## How it works (important details)

### Observations
- `preprocess_frame(frame)`:
  - Resizes frames to `84x84`, transposes to channel-first `(C,H,W)`, normalizes to `[0,1]`, returns a `torch.Tensor` (no batch dim).
  - If frames are RGBA, alpha is removed.

### Rewards & Terminals (`MazeEnv`)
- Camera thread detects RED, YELLOW, GREEN checkpoints (pixel-count thresholds) and increments `_latest_reward`. Sequence must be `None/green → red → yellow → green` for valid rewards.
  - Red: +10, Yellow: +10 (note: code comment shows +5 but implementation adds +10), Green: +10.
- Black patch detection in the bottom third triggers a terminal:
  - If robot moved forward ≥ 8 consecutive steps: progressive penalty `-45 + -2*(forwards-7)`
  - Fewer forwards: `-45`
  - If turning when black detected: `-40`
- `step(action)` returns `(processed_frame, reward, done, info)` where `info["reason"]` describes end reason or checkpoint.

### Agent & Training
- `Double_Agent` uses a ConvDQN backbone, Adam optimizer, replay memory (list-based), and an episode-based exponential epsilon decay:
  ```
  epsilon = eps_end + (eps_start - eps_end) * exp(-current_episode / eps_decay_episodes)
  ```
  Defaults: `eps_start=0.5, eps_end=0.05, eps_decay_episodes=400`.
- `train.py` forces exploitative behavior for the first 6 steps of each episode (temporary strong decay) as a training heuristic.
- The model and optimizer states are saved to `dqn_model.pth` after each episode.

---

## Hardware & wiring

- Motor GPIO pins (BCM numbers in `MotorController`):
  - Left motor: IN1 = **GPIO22**, IN2 = **GPIO23**  
  - Right motor: IN3 = **GPIO25**, IN4 = **GPIO24**
- Use a motor driver (e.g., H-bridge / L298N). Do **not** power motors directly from the Pi.
- Camera uses `picamera2` with `BGR888` preview resolution `(640,480)`.

---

## Troubleshooting

- **No frames / `reset()` timeout**: ensure picamera2 is installed/enabled and `Camera.get_frame()` returns a frame. `MazeEnv.reset()` waits up to 5s for a valid frame.
- **Shape mismatch**: confirm agent `obs_dim` (e.g., `(3,84,84)`) matches `preprocess_frame` output channels.
- **GPIO permission errors**: run with sudo or add your user to gpio group.
- **PyTorch on Pi**: use an appropriate pip wheel for your Pi model and OS.


---

## Recording & analysis

<p align="center">
  <span style="display:inline-block;margin:8px;text-align:center;">
    <a href="https://www.youtube.com/watch?v=JHTGzpdtKpA" target="_blank" rel="noopener">
      <img src="https://img.youtube.com/vi/JHTGzpdtKpA/0.jpg" alt="First trial — Episode 10" width="240" style="display:block;">
    </a>
    <a href="https://www.youtube.com/watch?v=JHTGzpdtKpA" target="_blank" rel="noopener" style="text-decoration:none;color:inherit;font-size:14px;">
      <div>First trial — Episode 10</div>
    </a>
  </span>

  <span style="display:inline-block;margin:8px;text-align:center;">
    <a href="https://www.youtube.com/watch?v=IBdh6dh6xiI" target="_blank" rel="noopener">
      <img src="https://img.youtube.com/vi/IBdh6dh6xiI/0.jpg" alt="Episode 300" width="240" style="display:block;">
    </a>
    <a href="https://www.youtube.com/watch?v=IBdh6dh6xiI" target="_blank" rel="noopener" style="text-decoration:none;color:inherit;font-size:14px;">
      <div>Episode 300</div>
    </a>
  </span>

  <span style="display:inline-block;margin:8px;text-align:center;">
    <a href="https://www.youtube.com/watch?v=oACCa2qsE4Y" target="_blank" rel="noopener">
      <img src="https://img.youtube.com/vi/oACCa2qsE4Y/0.jpg" alt="Episode 450" width="240" style="display:block;">
    </a>
    <a href="https://www.youtube.com/watch?v=oACCa2qsE4Y" target="_blank" rel="noopener" style="text-decoration:none;color:inherit;font-size:14px;">
      <div>Episode 450</div>
    </a>
  </span>

  <span style="display:inline-block;margin:8px;text-align:center;">
    <a href="https://www.youtube.com/watch?v=6Ak2ZThoBK4" target="_blank" rel="noopener">
      <img src="https://img.youtube.com/vi/6Ak2ZThoBK4/0.jpg" alt="Episode 600" width="240" style="display:block;">
    </a>
    <a href="https://www.youtube.com/watch?v=6Ak2ZThoBK4" target="_blank" rel="noopener" style="text-decoration:none;color:inherit;font-size:14px;">
      <div>Episode 600</div>
    </a>
  </span>
</p>


```

---

## Safety notes

- Test motors with wheels off the ground first to avoid unexpected movement.
- Keep a physical kill switch or be prepared to cut power if the robot behaves unsafely.
- Ensure correct motor driver wiring and separate power supply for motors.

---


