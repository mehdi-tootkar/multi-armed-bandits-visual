Multi-Armed Bandit Visualizer

This project provides an interactive web-based simulation of classical multi-armed bandit algorithms. It is designed as an educational tool for understanding exploration–exploitation trade-offs and for experimenting with algorithms such as Epsilon-Greedy, UCB1, and Thompson Sampling.

The application is built with Python and Streamlit. It allows users to configure the number of arms, select an algorithm, set the number of trials, and visualize how the algorithm learns over time. The included plots illustrate how arm-value estimates evolve, how rewards accumulate, and how frequently each arm is selected.

Features

Visual and interactive bandit simulation.

Supports Epsilon-Greedy, UCB1, and Thompson Sampling.

Adjustable parameters (arms, trials, exploration rate, random seed).

Real-time plots generated using Altair.

Reproducible results through fixed seeds.

Clean and modular code structure for customization and experimentation.

How It Works

The multi-armed bandit problem is a classical setup in reinforcement learning.
Each "arm" represents an action with an unknown reward probability.
The goal is to maximize accumulated rewards over time by choosing the best arm, despite not knowing which one it is beforehand.

The simulator generates a set of hidden true probabilities, and the chosen algorithm attempts to estimate them through repeated interaction. As the simulation progresses, the agent gradually identifies the optimal arm while balancing exploration and exploitation.

Visual Results

Use the Streamlit interface to run simulations and generate output visualizations.
Below are placeholders indicating where you can place your saved plots in this README.

True Probabilities vs Estimated Values

Show a bar chart comparing the true click-through rate of each arm with the algorithm's final estimates.

![True vs Estimated](results/true_vs_estimates.png)


Cumulative Average Reward

Show the cumulative average reward to illustrate how quickly the algorithm converges toward the optimal arm.

![Cumulative Reward](results/cumulative_reward.png)

Arm Selection Counts

Show a bar chart of how often each arm was selected throughout the simulation.

PLACE IMAGE HERE
![Arm Counts](results/arm_counts.png)