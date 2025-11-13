"""Streamlit application for visualising multi‑armed bandit algorithms.

This app lets you experiment interactively with classical multi‑armed
bandit algorithms (epsilon‑greedy, UCB1 and Thompson sampling).  You
can adjust the number of arms, the number of rounds and, for
epsilon‑greedy, the value of ε.  After running the simulation the app
displays charts of the true probabilities, the learned estimates,
cumulative reward and how often each arm was selected.

To launch the app:

```
streamlit run src/ui_app.py
```

Note: the `requirements.txt` file lists the necessary dependencies.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from bandit_env import MultiArmedBandit
from algorithms import EpsilonGreedy, UCB1, ThompsonSampling, BanditAlgorithm


def run_simulation(
    algo: BanditAlgorithm,
    env: MultiArmedBandit,
    n_rounds: int
) -> Tuple[List[int], List[int]]:
    """Run a multi‑armed bandit simulation.

    Parameters
    ----------
    algo : BanditAlgorithm
        The agent implementing the bandit algorithm.
    env : MultiArmedBandit
        The environment with fixed reward probabilities.
    n_rounds : int
        Number of pulls (time steps) to simulate.

    Returns
    -------
    choices : list[int]
        The arm chosen at each step.
    rewards : list[int]
        The reward received at each step (0 or 1).
    """
    choices: List[int] = []
    rewards: List[int] = []
    for t in range(1, n_rounds + 1):
        arm = algo.select_arm(t)
        reward = env.pull(arm)
        algo.update(arm, reward)
        choices.append(arm)
        rewards.append(reward)
    return choices, rewards


def main() -> None:
    st.set_page_config(page_title="Bandit Visualizer", layout="centered")
    st.title("Multi‑Armed Bandit Visualizer")
    st.write(
        "Interactively explore classical multi‑armed bandit algorithms. "
        "Adjust the number of arms and trials, choose an algorithm, and "
        "see how quickly it learns which arm is best."
    )

    with st.sidebar:
        st.header("Configuration")
        n_arms = st.slider(
            "Number of arms",
            min_value=2,
            max_value=10,
            value=4,
            step=1,
            help="Each arm has an independent Bernoulli reward distribution."
        )
        algo_name = st.selectbox(
            "Algorithm",
            options=["Epsilon‑Greedy", "UCB1", "Thompson Sampling"],
            index=0,
        )
        n_rounds = st.number_input(
            "Number of trials",
            min_value=10,
            max_value=20000,
            value=500,
            step=10,
            help="The simulation runs for this many steps."
        )
        random_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=100000,
            value=42,
            step=1,
            help="Set the NumPy random seed for reproducibility."
        )
        epsilon_value = None
        if algo_name == "Epsilon‑Greedy":
            epsilon_value = st.slider(
                "ε (exploration rate)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Probability of exploring a random arm on each step."
            )
        run_button = st.button("Run Simulation", type="primary")

    if run_button:
        # set seed for reproducibility
        np.random.seed(int(random_seed))
        # generate random true probabilities for each arm (5%–95%)
        true_probs = np.random.uniform(0.05, 0.95, size=n_arms)
        env = MultiArmedBandit(true_probs)

        # instantiate chosen algorithm
        if algo_name == "Epsilon‑Greedy":
            assert epsilon_value is not None
            algo = EpsilonGreedy(n_arms=n_arms, epsilon=epsilon_value)
        elif algo_name == "UCB1":
            algo = UCB1(n_arms=n_arms)
        elif algo_name == "Thompson Sampling":
            algo = ThompsonSampling(n_arms=n_arms)
        else:
            st.error(f"Unknown algorithm {algo_name}")
            return

        # Run simulation
        choices, rewards = run_simulation(algo, env, int(n_rounds))

        # Convert to useful summaries
        cum_reward = np.cumsum(rewards)
        avg_reward = cum_reward / np.arange(1, int(n_rounds) + 1)
        counts = pd.Series(choices).value_counts().sort_index()

        # Compute estimated probabilities / values for final display
        if isinstance(algo, (EpsilonGreedy, UCB1)):
            estimates = algo.values.copy()
        elif isinstance(algo, ThompsonSampling):
            total = algo.successes + algo.failures
            estimates = np.divide(algo.successes, total, out=np.zeros_like(total), where=total > 0)
        else:
            estimates = np.zeros(n_arms)

        st.subheader("True probabilities vs. final estimates")
        comparison_df = pd.DataFrame({
            "True Probability": true_probs,
            "Estimated Value/Probability": estimates,
        })
        st.bar_chart(comparison_df)

        st.subheader("Cumulative average reward")
        # show a line chart of average reward across trials
        avg_df = pd.DataFrame({"Average Reward": avg_reward})
        st.line_chart(avg_df)

        st.subheader("Arm selection counts")
        # Ensure counts include all arms (set missing counts to zero)
        all_counts = counts.reindex(range(n_arms), fill_value=0)
        counts_df = pd.DataFrame({"Counts": all_counts})
        st.bar_chart(counts_df)

        # optionally display underlying data in expanders
        with st.expander("Simulation details"):
            st.write("First 20 choices and rewards:")
            details = pd.DataFrame({"arm_chosen": choices[:20], "reward": rewards[:20]})
            st.dataframe(details)
            st.write("\nEstimated values/probabilities per arm:")
            st.dataframe(comparison_df.T)

    # Footer
    st.markdown(
        """
        ---
        *Adjust the settings on the
        left and click **Run Simulation** to see how different bandit
        algorithms perform.*
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
