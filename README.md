Multi-Armed Bandit Visualizer

This project provides an interactive web-based simulation of classical multi-armed bandit algorithms. It is designed as an educational tool for understanding exploration–exploitation trade-offs and for experimenting with algorithms such as Epsilon-Greedy, UCB1, and Thompson Sampling.

The application is built with Python and Streamlit. It allows users to configure the number of arms, select an algorithm, set the number of trials, and visualize how the algorithm learns over time. The included plots illustrate how arm-value estimates evolve, how rewards accumulate, and how frequently each arm is selected.

Visual Results

Use the Streamlit interface to run simulations and generate output visualizations.
Below are placeholders indicating where you can place your saved plots in this README.

True Probabilities vs Estimated Values

![True vs Estimated](results/true_vs_estimates.png)

Show a bar chart comparing the true click-through rate of each arm with the algorithm's final estimates.



Cumulative Average Reward

![Cumulative Reward](results/cumulative_reward.png)

Show the cumulative average reward to illustrate how quickly the algorithm converges toward the optimal arm.


Arm Selection Counts

![Arm Counts](results/arm_counts.png)

Show a bar chart of how often each arm was selected throughout the simulation.

