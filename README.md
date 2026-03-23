# Intelligent Omi: Cooperative Multi-Agent Reinforcement Learning

Welcome to the **Intelligent Omi** project! This is an advanced artificial intelligence environment designed to teach agents how to master the Sri Lankan trick-taking card game, **Omi**, through deep reinforcement learning.

## 🌟 Project Vision
The goal of this project is to build a cooperative AI that learns complex strategy, teamwork, and card-counting without any hard-coded rules for strategy. Using **Deep Reinforcement Learning (MAPPO)** and **Long Short-Term Memory (LSTM)**, the agents learn from purely randomized play to become expert Omi players over millions of matches.

---

## 🛠 Features

*   **PettingZoo AEC Environment**: A strict, turn-based referee system that enforces Omi rules (must-follow-suit, trump hierarchy).
*   **CTDE Architecture**: "Centralized Training, Decentralized Execution." The AI is trained by an omniscient critic, but plays fair matches during execution using only its private hand and memory.
*   **Recurrent Brains (LSTM)**: Each agent has an LSTM memory, allowing it to "count cards" by remembering what has been played earlier in the match.
*   **Action Masking**: The AI is physically blocked from making illegal moves, focusing 100% of its power on strategy.
*   **Dense Reward System**: A toggleable, high-frequency feedback system for lightning-fast training.

---

## 📁 Repository Structure

### ⚖️ The Game Engine (`omi_env/`)
*   `rules.py`: The core Omi logic (shuffling, dealing, trick resolution, legal move masking).
*   `env.py`: The "Referee" wrapper. Manages turns, rewards, and the current game stage (Trump vs. Play).
*   `encoding.py`: Translates cards and game states into numbers for the AI's neural network.

### 🧠 The AI Brains (`models/`)
*   `policy.py` (The Actor): Each player's decentralized brain. Uses an LSTM to track history.
*   `critic.py` (The Critic): The omniscient coach used only during training to grade plays.

### 🎓 Training Loop (`marl/` & `scripts/`)
*   `r_mappo.py`: The math behind the policy updates (Proximal Policy Optimization).
*   `train.py`: The main script to start a training session.
*   `eval.py`: Compare your trained AI against random or rule-based bots.
*   `inference/`: Minimal scripts to run a single trained model for testing or demos.

---

## 🏆 Improved Rewarding System
This project features a high-density reward system that can be toggled in your config.

*   **Trick Rewards (+0.1)**: Immediate feedback for winning a trick as a team.
*   **Illegal Move Penalty (-0.1)**: Teaches the AI the game rules faster.
*   **Over-playing Penalty (-0.05)**: Discourages wasting high cards when a teammate is already winning the trick. **Only triggers if a choice was available!**
*   **Margin-based Final Wins**: Rewards are scaled by the margin of victory (e.g., winning 7-1 is better than 5-3).
*   **Trump Declarer Bonus (+0.1)**: Incentivizes the declarer to call the most effective trump suit.

> [!TIP]
> Toggle this in your configuration using `reward_shaping: true`.

---

## 🚀 Quickstart Guide

### 1️⃣ Installation
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

### 2️⃣ Run Unit Tests
Verify the environment and rules are working perfectly.
```bash
pytest
```

### 3️⃣ Train Your AI
Run a fast CPU-based demo:
```powershell
python scripts/train.py --config configs/small.yaml
```
Run the full training session (default):
```powershell
python scripts/train.py --config configs/default.yaml
```

### 4️⃣ Evaluate Performance
Once trained, compare your AI's win rate against a rule-based baseline:
```powershell
python scripts/eval.py --config configs/default.yaml --weights runs/default_cpu/policy_last.pt
```

---

## ⚙️ Configuration
You can customize training speed, model size, and rewarding in `configs/default.yaml`.
*   Change `episodes` to train for longer.
*   Adjust `lr` (learning rate) for faster or more stable convergence.
*   Set `reward_shaping: true` to use the improved dense feedback system.

---

## 🛡️ License
Designed for researchers and Omi enthusiasts. Built with PettingZoo and PyTorch.
