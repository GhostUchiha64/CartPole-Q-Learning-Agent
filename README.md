# CartPole Q-Learning Agent

**DATA 530: Artificial Intelligence — Final Project** | Siddartha Bandi

---

## Overview

This project implements a **tabular Q-Learning agent** to solve the classic `CartPole-v1` reinforcement learning environment from the [Gymnasium](https://gymnasium.farama.org/) library. The agent learns to balance a pole on a moving cart by discretizing the continuous 4-dimensional state space into bins and updating a Q-table using the **epsilon-greedy exploration strategy**.

The project demonstrates core Reinforcement Learning concepts: the Bellman equation, temporal-difference (TD) learning, exploration vs. exploitation tradeoff, and state-space discretization for tabular RL methods.

---

## Problem Statement

In `CartPole-v1`, an agent must keep a pole balanced upright on a cart by applying left or right force at each timestep. The episode ends when the pole angle exceeds ±12° or the cart moves ±2.4 units from center. The goal is to achieve an average reward of 200 (maximum per episode) consistently.

---

## Methodology

### State Space Discretization
The continuous state has 4 variables, discretized into bins:

| State Variable | Bins | Range |
|---------------|------|-------|
| Cart Position | 3 | [-2.4, 2.4] |
| Cart Velocity | 3 | [-3.0, 3.0] |
| Pole Angle | 6 | [-0.418, 0.418] |
| Pole Velocity | 12 | [-3.0, 3.0] |

**Total Q-Table States:** 3 × 3 × 6 × 12 = **648 discrete states** × 2 actions = 1,296 entries

### Q-Learning Algorithm

```
Q(s, a) ← Q(s, a) + α × [r + γ × max Q(s', a') - Q(s, a)]
```

| Hyperparameter | Value | Description |
|---------------|-------|-------------|
| `alpha` (α) | 0.1 | Learning rate |
| `gamma` (γ) | 0.99 | Discount factor |
| `epsilon` (ε) | 1.0 → 0.01 | Exploration rate (decays by 0.995/episode) |
| Episodes | 1,000 | Total training episodes |
| Max Steps | 200 | Max steps per episode |
| Random Seed | 42 | For reproducibility |

### Training Loop
1. Reset environment → get initial state
2. Discretize state → look up Q-table
3. Select action (ε-greedy: explore or exploit)
4. Execute action → receive reward, next state
5. Update Q-table via TD learning
6. Decay epsilon after each episode
7. Log average reward every 100 episodes

---

## Results

- The agent learns to balance the pole progressively over 1,000 episodes
- The **50-episode moving average reward** stabilizes and trends upward, demonstrating convergence
- A `training_curve.png` is generated automatically showing Total Reward per Episode + Moving Average trendline
- Simulation video recorded using `gymnasium.wrappers.RecordVideo`

---

## Repository Contents

```
Siddartha Bandi_Final Project_Deliverables/
├── README.md
├── siddartha_bandi_final_project_q_learning_agent.py    # Main Q-Learning agent script
├── ReadMe.txt                                            # Original setup instructions
├── Siddartha Bandi Final Project Report (Q-Learning Agent).pdf   # Final report
└── Siddartha Bandi Final Project Report (Q-Learning Agent).docx  # Report (Word)
```

> **Note:** Video recording (`.mp4`), audio (`.m4a`), Zoom meeting folder, and `.zip` archive are excluded due to file size.

---

## Setup & Usage

### Install Dependencies
```bash
pip install gymnasium numpy matplotlib
```

### Run the Agent
```bash
python siddartha_bandi_final_project_q_learning_agent.py
```

### Expected Output
```
STATE VARIABLE       | BINS  | LOWER    | UPPER
------------------------------------------------------------
Cart Position        | 3     | -2.4     | 2.4
Cart Velocity        | 3     | -3.0     | 3.0
Pole Angle           | 6     | -0.418   | 0.418
Pole Velocity        | 12    | -3.0     | 3.0

Starting Training for 1000 episodes...
Episode 100/1000 | Avg Reward: XX.XX | Epsilon: 0.XXXX
...
Episode 1000/1000 | Avg Reward: XX.XX | Epsilon: 0.0100

Training curve saved as 'training_curve.png'
```

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| RL Environment | Gymnasium (`CartPole-v1`) |
| Numerical Computing | NumPy |
| Visualization | Matplotlib |
| Environment | Google Colab / Local Python |

---

## Key Concepts Demonstrated

- **Temporal Difference (TD) Learning** — updating Q-values from bootstrapped estimates
- **Epsilon-Greedy Exploration** — balancing exploration and exploitation
- **State Discretization** — converting continuous spaces for tabular Q-learning
- **Convergence Analysis** — visualizing learning stability via moving average rewards

---

## Author

**Siddartha Bandi**
DATA 530: Artificial Intelligence — Final Project
