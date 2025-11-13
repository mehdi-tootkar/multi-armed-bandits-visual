"""Implementations of classical multi‑armed bandit algorithms.

This module contains three simple algorithms for the multi‑armed bandit
problem: epsilon‑greedy, UCB1 and Thompson Sampling.  Each class
maintains its own internal state and exposes a consistent API with
`select_arm(t)` and `update(arm, reward)` methods.  The optional
argument `t` passed to `select_arm` gives the current time step,
allowing algorithms such as UCB1 to compute confidence bounds.
"""

from __future__ import annotations

import numpy as np


class BanditAlgorithm:
    """Abstract base class for bandit algorithms.

    Concrete subclasses must implement `select_arm` and `update`.  The
    base class stores the number of arms for convenience.
    """

    def __init__(self, n_arms: int) -> None:
        self.n_arms = int(n_arms)
        if self.n_arms <= 0:
            raise ValueError("number of arms must be positive")

    def select_arm(self, t: int) -> int:
        """Return the index of the arm to pull at time step `t`.

        Parameters
        ----------
        t : int
            Current time step (starting from 1).  Some algorithms
            ignore this parameter (e.g. epsilon‑greedy), but others
            (e.g. UCB) require it to compute their exploration bonus.
        """
        raise NotImplementedError

    def update(self, arm: int, reward: int) -> None:
        """Update the algorithm's internal state after pulling `arm` and
        receiving `reward`.  Should be called after every round.
        """
        raise NotImplementedError


class EpsilonGreedy(BanditAlgorithm):
    """Classic epsilon‑greedy algorithm.

    With probability `epsilon`, the agent chooses a random arm (explore);
    otherwise it chooses the arm with the highest estimated value
    (exploit).  The estimated value for each arm is the running
    average of observed rewards.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1) -> None:
        super().__init__(n_arms)
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        self.epsilon = float(epsilon)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)

    def select_arm(self, t: int | None = None) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: int) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1(BanditAlgorithm):
    """Upper Confidence Bound (UCB1) algorithm.

    UCB1 maintains an optimistic estimate of the value of each arm by
    adding an exploration bonus to the empirical mean.  Arms with
    fewer pulls get a larger bonus.  This algorithm has strong
    theoretical guarantees of performance under stationary reward
    distributions.
    """

    def __init__(self, n_arms: int) -> None:
        super().__init__(n_arms)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.values = np.zeros(self.n_arms, dtype=float)

    def select_arm(self, t: int) -> int:
        # first pull each arm at least once to initialise estimates
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # compute UCB value: mean + exploration bonus
        bonus = np.sqrt(2 * np.log(t) / self.counts)
        ucb_values = self.values + bonus
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: int) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        # incremental mean update
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSampling(BanditAlgorithm):
    """Thompson Sampling for Bernoulli bandits.

    Maintains Beta distributions for each arm's success probability.  On
    each round it samples a candidate probability for each arm and
    chooses the arm with the largest sample.  After observing a reward
    the Beta parameters are updated.
    """

    def __init__(self, n_arms: int) -> None:
        super().__init__(n_arms)
        self.successes = np.zeros(self.n_arms, dtype=float)
        self.failures = np.zeros(self.n_arms, dtype=float)

    def select_arm(self, t: int | None = None) -> int:
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: int) -> None:
        # reward is assumed to be 0 or 1
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1 for Bernoulli bandits")
        if reward == 1:
            self.successes[arm] += 1.0
        else:
            self.failures[arm] += 1.0
