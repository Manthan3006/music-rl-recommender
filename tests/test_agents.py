"""LinUCB's incremental (Sherman-Morrison) A_inv must equal a direct inverse
of A = I + sum x x^T, and UCB1's counts / step counter must persist."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.rl.agents import UCB1, LinUCB


def test_linucb_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(0)
    dim, n_updates = 30, 500
    agent = LinUCB(dim=dim, alpha=1.0, seed=0)

    A = np.eye(dim)
    b = np.zeros(dim)
    for _ in range(n_updates):
        x = rng.normal(size=dim)
        reward = float(rng.normal())
        agent._last_context = x
        agent.update(None, 0, reward, None, False)
        A += np.outer(x, x)
        b += reward * x

    np.testing.assert_allclose(agent.A_inv, np.linalg.inv(A), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(agent.A_inv @ b, np.linalg.solve(A, b), rtol=1e-8, atol=1e-10)


def test_ucb1_counts_persist_and_every_arm_tried_first():
    agent = UCB1(n_arms=10, seed=0)
    first_ten = []
    for _ in range(10):
        a = agent.select_action(None)
        first_ten.append(a)
        agent.update(None, a, 1.0, None, False)
    assert sorted(first_ten) == list(range(10))  # every arm pulled once before the UCB rule

    for _ in range(90):
        a = agent.select_action(None)
        agent.update(None, a, 1.0, None, False)
    assert agent.t == 100 and agent.counts.sum() == 100
