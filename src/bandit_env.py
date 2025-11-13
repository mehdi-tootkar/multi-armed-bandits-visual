"""Definition of a simple multi‑armed bandit environment.

This module defines a `MultiArmedBandit` class that encapsulates the
underlying click‑through probabilities for each arm and provides a
`pull` method to simulate drawing an arm and receiving a Bernoulli
reward.  It is deliberately minimal so that the learning agents in
`algorithms.py` can focus on their own update logic.
"""

from __future__ import annotations

import numpy as np


class MultiArmedBandit:
    """Simple k‑armed bandit environment.

    Parameters
    ----------
    probs : array_like
        A 1‑D array of length `k` where each entry is the true
        probability of receiving a reward of 1 when pulling that arm.

    Notes
    -----
    The bandit is stationary: the reward probabilities do not change
    over time.  The reward is binary (0 or 1).  A non‑stationary
    version (e.g. with drifting probabilities) can be added in the
    future if needed.
    """

    def __init__(self, probs: np.ndarray | list[float]) -> None:
        self.probs = np.asarray(probs, dtype=float)
        if self.probs.ndim != 1:
            raise ValueError("probabilities must be a 1‑D array")
        if np.any((self.probs < 0) | (self.probs > 1)):
            raise ValueError("all probabilities must be between 0 and 1")

    @property
    def n_arms(self) -> int:
        """Return the number of arms."""
        return self.probs.size

    def pull(self, arm: int) -> int:
        """Simulate pulling the specified arm and return a reward.

        The reward is sampled from a Bernoulli distribution with the
        arm's true probability.

        Parameters
        ----------
        arm : int
            Index of the arm to pull (0‑based).

        Returns
        -------
        int
            1 if success; 0 otherwise.
        """
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} out of bounds for {self.n_arms} arms")
        return int(np.random.random() < self.probs[arm])
