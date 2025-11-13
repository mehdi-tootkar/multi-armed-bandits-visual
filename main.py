"""Command‑line runner for the multi‑armed bandit algorithms.

This script allows you to run a bandit simulation from the terminal
without using the Streamlit UI.  It generates random true
probabilities for each arm, executes the chosen algorithm for a
specified number of rounds, and prints summary statistics.

Usage example:

```
python main.py --arms 5 --rounds 1000 --algo ucb
```
"""

from __future__ import annotations

import argparse
import numpy as np

from src.bandit_env import MultiArmedBandit
from src.algorithms import EpsilonGreedy, UCB1, ThompsonSampling
def run_simulation(algo, env, n_rounds):
    """Run a multi‑armed bandit simulation.

    This internal helper function duplicates the logic from the
    Streamlit app but avoids importing Streamlit when running from
    the command line.  It returns the chosen arms and observed
    rewards over `n_rounds` steps.
    """
    choices = []
    rewards = []
    for t in range(1, n_rounds + 1):
        arm = algo.select_arm(t)
        reward = env.pull(arm)
        algo.update(arm, reward)
        choices.append(arm)
        rewards.append(reward)
    return choices, rewards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi‑armed bandit simulation.")
    parser.add_argument("--arms", type=int, default=4, help="Number of arms")
    parser.add_argument("--rounds", type=int, default=500, help="Number of trials")
    parser.add_argument("--algo", type=str, default="epsilon", choices=["epsilon", "ucb", "thompson"],
                        help="Which algorithm to run (epsilon | ucb | thompson)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate for epsilon‑greedy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    true_probs = np.random.uniform(0.05, 0.95, size=args.arms)
    env = MultiArmedBandit(true_probs)
    if args.algo == "epsilon":
        algo = EpsilonGreedy(n_arms=args.arms, epsilon=args.epsilon)
    elif args.algo == "ucb":
        algo = UCB1(n_arms=args.arms)
    else:
        algo = ThompsonSampling(n_arms=args.arms)
    choices, rewards = run_simulation(algo, env, args.rounds)
    cum_reward = np.cumsum(rewards)
    avg_reward = cum_reward / np.arange(1, args.rounds + 1)
    # compute final estimates
    if isinstance(algo, (EpsilonGreedy, UCB1)):
        estimates = algo.values
    else:
        total = algo.successes + algo.failures
        estimates = np.divide(algo.successes, total, out=np.zeros_like(total), where=total > 0)
    print(f"True probabilities: {true_probs}")
    print(f"Estimated values/probabilities: {estimates}")
    print(f"Average reward over all rounds: {avg_reward[-1]:.4f}")
    # print how many times each arm was chosen
    counts = np.bincount(np.array(choices), minlength=args.arms)
    print(f"Arm selection counts: {counts}")


if __name__ == "__main__":
    main()