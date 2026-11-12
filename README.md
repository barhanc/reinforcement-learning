# Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29+-000000.svg?style=flat)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A practical and rigorous curriculum covering modern Reinforcement Learning algorithms and paradigms. From classical multi-armed bandits and tabular temporal-difference control to deep continuous actor-critics, hindsight goal-conditioning, offline Decision Transformers, and LLM alignment with Group Relative Policy Optimization (GRPO).

## Overview

<details open>
<summary><b>Lab 1: Multi-Armed Bernoulli Bandits</b></summary>

- **Focus**: The fundamental exploration vs. exploitation dilemma in stationary stochastic reward settings.
- **Key Concepts**:
  - Implementation of heuristic and principled exploration strategies: $\epsilon$-greedy, optimistic initial values, and Upper Confidence Bound (UCB1).
  - Bayesian bandit formulation using **Thompson Sampling** with conjugate Beta-Bernoulli priors.
  - Empirical cumulative regret evaluation against the theoretical Lai-Robbins logarithmic lower bound.
- **Directory**: [`lab01/`](./lab01/)
</details>

<details>
<summary><b>Lab 2: Tabular TD Learning — SARSA & Racetracks</b></summary>

- **Focus**: Model-free prediction and control in discrete Markov Decision Processes (MDPs).
- **Key Concepts**:
  - Temporal Difference control using off-policy $n$-step **SARSA**.
  - State representation for physical kinematics: 2D position and discrete velocity vectors with acceleration and braking dynamics.
  - Handling terminal track boundaries, track collision resets, and convergence to near-optimal navigation policies on racetrack grids.
- **Directory**: [`lab02/`](./lab02/)
</details>

<details>
<summary><b>Lab 3: Monte Carlo Tree Search (MCTS) & Planning</b></summary>

- **Focus**: Tree search, heuristic lookahead, and adversarial self-play.
- **Key Concepts**:
  - The 4 core stages of MCTS: **Selection** (via UCT - Upper Confidence Bounds for Trees), **Expansion**, **Simulation** (Monte Carlo rollout), and **Backpropagation**.
  - Balancing tree exploration vs. deep exploitation in zero-sum game trees.
  - Evaluation and benchmarking on the strategic board game **Isolation**.
- **Directory**: [`lab03/`](./lab03/)
</details>

<details>
<summary><b>Lab 4: Advantage Actor-Critic (A2C) for Lunar Lander</b></summary>

- **Focus**: Policy gradient methods, variance reduction, and neural value baselines.
- **Key Concepts**:
  - Decoupled / shared parameterized architectures: Policy network (Actor) predicting action logits/Gaussian distributions and Value network (Critic) estimating $V^\pi(s)$.
  - Generalized Advantage Estimation (GAE) and advantage computation $A(s, a) = R + \gamma V(s') - V(s)$ for stabilized policy gradient updates.
  - Entropy regularization bonus to prevent premature policy collapse.
  - Solving the continuous/discrete **Gymnasium LunarLander-v2** environment.
- **Directory**: [`lab04/`](./lab04/)
</details>

<details>
<summary><b>Lab 5: LLM Alignment with GRPO (Group Relative Policy Optimization)</b></summary>

- **Focus**: Post-training and reinforcement learning alignment for Large Language Models.
- **Key Concepts**:
  - The mechanics of **GRPO** (as popularized by DeepSeekMath and DeepSeek-R1): Eliminating the need for a separate value/critic network by sampling groups of completions for each prompt.
  - Computing group-relative baseline rewards: Standardizing reward scores across completion groups to compute policy advantages.
  - Integration with **Hugging Face TRL** (`GRPOTrainer`) and causal language models for targeted reward criteria (e.g. prompt compliance, vowel constraints, formatting).
- **Directory**: [`lab05/`](./lab05/)
</details>

<details>
<summary><b>Lab 6: Soft Actor-Critic (SAC) + Hindsight Experience Replay (HER)</b></summary>

- **Focus**: Off-policy continuous deep RL, entropy maximization, and learning from sparse binary failure signals.
- **Key Concepts**:
  - **Soft Actor-Critic (SAC)**: Maximum entropy objective encouraging diverse exploration, twin $Q$-networks to mitigate overestimation bias, and automated temperature $\alpha$ adjustment.
  - **Hindsight Experience Replay (HER)**: Re-labeling failed trajectories with virtual goals achieved in hindsight (using the `'future'` sampling strategy).
  - Training on robotic manipulation tasks via **Panda-Gym** (`PandaReach-v3`) with sparse multi-dimensional reward signals.
- **Directory**: [`lab06/`](./lab06/)
</details>

<details>
<summary><b>Lab 7: Offline Reinforcement Learning — Decision Transformer (nanoDT)</b></summary>

- **Focus**: Treating reinforcement learning as a conditional sequence modeling task without dynamic programming.
- **Key Concepts**:
  - The **Decision Transformer** paradigm: Autoregressively modeling tokenized trajectories of returns-to-go, states, and actions $(\hat{R}_t, s_t, a_t)$ via causal self-attention.
  - Expert dataset generation: Collecting rollouts with trained SAC policies and processing offline demonstration datasets via **Minari**.
  - Conditioning test-time rollouts on desired target returns to achieve reward-maximizing closed-loop control.
- **Directory**: [`lab07/`](./lab07/)
</details>

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/barhanc/reinforcement-learning.git
cd reinforcement-learning
```

### 2. Environment Setup

Each module can be run with standard Python environments (or managed with [uv](https://docs.astral.sh/uv/)):

```bash
# Create and activate an environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install core dependencies
pip install torch gymnasium numpy scipy matplotlib
```

Individual lab directories (e.g., `lab04/`, `lab06/`) contain task-specific `requirements.txt` files for specialized dependencies such as `panda-gym`, `trl`, or `minari`.
