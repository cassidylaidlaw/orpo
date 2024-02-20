from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np
import tqdm
from scipy.sparse import csr_matrix


def load_mdp(transition_path, reward_path) -> Tuple[np.ndarray, np.ndarray]:
    transitions = np.load(transition_path)
    rewards = np.load(reward_path)

    num_states, num_actions = transitions.shape
    done_state = num_states
    num_states += 1
    transitions = np.concatenate([transitions, np.zeros((1, num_actions))])
    transitions[transitions == -1] = done_state
    transitions[done_state, :] = done_state
    rewards = np.concatenate([rewards, np.zeros((1, num_actions))])
    return transitions, rewards


def get_sparse_mdp(
    transitions: np.ndarray, rewards: np.ndarray
) -> Tuple[csr_matrix, np.ndarray]:
    num_states, num_actions = transitions.shape
    num_state_actions = num_states * num_actions
    sparse_transitions = csr_matrix(
        (
            np.ones(num_state_actions),
            (np.arange(num_state_actions, dtype=int), transitions.ravel()),
        ),
        shape=(num_state_actions, num_states),
        dtype=np.float32,
    )
    rewards_vector = rewards.ravel().astype(np.float32)
    return sparse_transitions, rewards_vector


@dataclass
class ValueIterationResults(object):
    random_qs: np.ndarray
    random_values: np.ndarray
    optimal_qs: np.ndarray
    optimal_values: np.ndarray
    worst_qs: np.ndarray
    worst_values: np.ndarray


def run_value_iteration(
    sparse_transitions: csr_matrix, rewards_vector: np.ndarray, horizon: int
) -> ValueIterationResults:
    num_state_actions, num_states = cast(Tuple[int, int], sparse_transitions.shape)
    num_actions = num_state_actions // num_states

    done_q = np.zeros((num_states, num_actions), dtype=rewards_vector.dtype)
    done_v = np.zeros(num_states, dtype=rewards_vector.dtype)

    random_qs: List[np.ndarray] = [done_q]
    random_values: List[np.ndarray] = [done_v]
    optimal_qs: List[np.ndarray] = [done_q]
    optimal_values: List[np.ndarray] = [done_v]
    worst_qs: List[np.ndarray] = [done_q]
    worst_values: List[np.ndarray] = [done_v]

    for t in tqdm.tqdm(list(reversed(list(range(horizon)))), desc="Value iteration"):
        random_qs.insert(
            0,
            (rewards_vector + sparse_transitions @ random_values[0]).reshape(
                (num_states, num_actions)
            ),
        )
        random_values.insert(0, random_qs[0].mean(axis=1))

        optimal_qs.insert(
            0,
            (rewards_vector + sparse_transitions @ optimal_values[0]).reshape(
                (num_states, num_actions)
            ),
        )
        optimal_values.insert(0, optimal_qs[0].max(axis=1))

        worst_qs.insert(
            0,
            (rewards_vector + sparse_transitions @ worst_values[0]).reshape(
                (num_states, num_actions)
            ),
        )
        worst_values.insert(0, worst_qs[0].min(axis=1))

    return ValueIterationResults(
        random_qs=np.array(random_qs[:-1]),
        random_values=np.array(random_values[:-1]),
        optimal_qs=np.array(optimal_qs[:-1]),
        optimal_values=np.array(optimal_values[:-1]),
        worst_qs=np.array(worst_qs[:-1]),
        worst_values=np.array(worst_values[:-1]),
    )


if __name__ == "__main__":
    transitions, rewards = load_mdp("transitions.npy", "rewards.npy")
    sparse_transitions, rewards_vector = get_sparse_mdp(transitions, rewards)
    q_values = run_value_iteration(sparse_transitions, rewards_vector, 10)
