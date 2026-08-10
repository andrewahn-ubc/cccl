import numpy as np

from src.schedules import DecayedTransitions, discounted_occupancy, oracle_transition


def test_transition_rows_decay_and_oracle():
    oracle = oracle_transition((0.7, 0.3))
    assert np.allclose(oracle.sum(1), 1)
    transitions = DecayedTransitions(n_tasks=2, alpha=0.5, decay=0.5)
    transitions.update(0, 1)
    transitions.update(1, 0)
    assert np.isclose(transitions.counts[0, 1], 0.5)
    assert np.isclose(transitions.counts[1, 0], 1.0)
    assert np.allclose(transitions.matrix().sum(1), 1)


def test_discounted_occupancy_matches_two_state_chain():
    transition = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    # From state zero: h=1 is state one, h=2 is state zero.
    result = discounted_occupancy(transition, current=0, horizon=2, gamma=0.5)
    assert np.allclose(result, [1 / 3, 2 / 3])

