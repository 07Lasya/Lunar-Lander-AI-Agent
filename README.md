# 🚀 Lunar Lander AI Agent

An AI agent developed using a hybrid of **Particle Swarm Optimization (PSO)** and **Tabu Search** to solve the classic OpenAI Lunar Lander environment.

## 🎯 Objective
To land the lunar module safely using an optimized policy derived from algorithmic search and reinforcement learning techniques.

## 🧠 Features
- Implements a softmax policy on top of a linear model
- Parameter file (`best_params.npy`) stores optimized weights and biases
- Successfully simulated over **1000 episodes** with an **87% landing success rate**

## 🧩 Files
- `policy_2101.py`: Main agent logic to select actions based on policy
- `best_params.npy`: Optimized weights and biases used by the policy

## 🛠️ Tech Stack
- Python
- NumPy
- PSO & Tabu Search (offline optimization logic)

## 🚀 How to Use
```python
import numpy as np
from policy_2101 import policy_action

params = np.load("best_params.npy", allow_pickle=True)
observation = env.reset()
action = policy_action(params, observation)
