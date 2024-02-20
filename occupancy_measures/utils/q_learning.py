import itertools
import os
from typing import Tuple

import numpy as np
import tqdm
from matplotlib import pyplot as plt

from occupancy_measures.utils.mdp_utils import get_sparse_mdp

path = "./"


def calculate_occupancy_measure_TV(sa_occ: np.ndarray, num_policies: int):
    TV_distances = np.abs(sa_occ[np.newaxis, :] - sa_occ[:, np.newaxis]).sum(
        axis=(2, 3)
    )
    return TV_distances


def calculate_action_dist_KL(sa_occ: np.ndarray, policies: np.ndarray) -> np.ndarray:
    num_policies = policies.shape[0]
    KL_distances = np.zeros((num_policies, num_policies))
    for p1, p2 in itertools.combinations(np.arange(num_policies), 2):
        KL_distances[p1, p2] = (
            sa_occ[p1] * (np.log(policies[p1]) - np.log(policies[p2]))
        ).sum()
        KL_distances[p2, p1] = (
            sa_occ[p2] * (np.log(policies[p2]) - np.log(policies[p1]))
        ).sum()

    return KL_distances


def calculate_occupancy_measure(
    policy: np.ndarray,
    sparse_transitions: np.ndarray,
    num_states: int,
    num_actions: int,
    horizon: int = 100,
    gamma: float = 0.99,
) -> Tuple[np.ndarray, np.ndarray]:
    state_occupancy = np.zeros((num_states))
    state_occupancy[0] = 1
    per_time_policy_state_action_occupancy = np.zeros(
        (horizon, num_states, num_actions)
    )
    policy_state_action_occupancy = np.zeros((num_states, num_actions))

    for step in range(horizon):
        state_action_occupancy = state_occupancy[:, None] * policy[step]
        policy_state_action_occupancy = policy_state_action_occupancy + (
            state_action_occupancy * (gamma**step)
        )
        per_time_policy_state_action_occupancy[step] = state_action_occupancy
        state_occupancy = sparse_transitions.T @ state_action_occupancy.ravel().astype(
            np.float32
        )

    return policy_state_action_occupancy, per_time_policy_state_action_occupancy


def run_tabular_q_learning(
    transitions: np.ndarray,
    rewards: np.ndarray,
    num_states: int,
    num_actions: int,
    max_epsilon: float,
    min_epsilon: float,
    epsilon_for_computing_policies: float,
    num_episodes: int = 10000,
    checkpoint: int = 100,
    horizon: int = 100,
    alpha: float = 0.1,
    gamma: float = 0.99,
) -> np.ndarray:
    epsilon = max_epsilon
    q_table = np.zeros((horizon, num_states, num_actions))
    policies = np.zeros((num_episodes // checkpoint, horizon, num_states, num_actions))
    num_policies = 0

    for episode in tqdm.tqdm(range(num_episodes)):
        state = 0
        action = 0  # type: int
        for step in range(horizon):
            if np.random.uniform(0, 1) <= epsilon:
                action = np.random.randint(0, num_actions)
            else:
                action = np.argmax(q_table[step, state, :]).astype(int)

            next_state = int(transitions[state, action])
            reward = rewards[state, action]

            old_val = (1 - alpha) * q_table[step, state, action]
            if (step + 1) < horizon:
                new_val = alpha * (
                    reward + (gamma * np.max(q_table[step + 1, next_state, :]))
                )
            else:
                new_val = alpha * reward

            q_table[step, state, action] = old_val + new_val

            if episode % checkpoint == 0:
                # Epsilon greedy
                policies[
                    num_policies,
                    step,
                    q_table[step] == q_table[step].max(axis=1)[:, None],
                ] = 1

                policies[num_policies, step] *= (
                    1 - epsilon_for_computing_policies
                ) / policies[num_policies, step].sum(axis=-1)[..., None]
                policies[num_policies, step, :] += (
                    epsilon_for_computing_policies / num_actions
                )

                # Boltzman exploration
                # policies[num_policies, step] = np.exp(q_table[step]/epsilon)
                # policies[num_policies, step] /= np.sum(policies[num_policies, step])

            state = next_state

        if episode % checkpoint == 0:
            num_policies += 1

        epsilon = max_epsilon - episode / num_episodes * (max_epsilon - min_epsilon)

    return policies


def generate_plots(
    filepath: str,
    level: int,
    steps: int,
    min_epsilon: float = 1.0,
    max_epsilon: float = 1.0,
):
    data_filepath = filepath + "saved_matrices"
    plot_filepath = filepath + "plots"

    assert os.path.exists(path + data_filepath), "Please build the matrices first!"

    if not os.path.exists(path + plot_filepath):
        os.makedirs(path + plot_filepath)

    transitions = np.load(data_filepath + "/transitions_" + str(level) + ".npy")
    proxy_rewards = np.load(data_filepath + "/proxy_rewards_" + str(level) + ".npy")
    true_rewards = np.load(data_filepath + "/true_rewards_" + str(level) + ".npy")
    sparse_transitions, _ = get_sparse_mdp(transitions, proxy_rewards)

    num_states, num_actions = transitions.shape

    policies = run_tabular_q_learning(
        transitions,
        proxy_rewards,
        num_states,
        num_actions,
        num_episodes=10000,
        epsilon_for_computing_policies=0.1,
        max_epsilon=min_epsilon,
        min_epsilon=max_epsilon,
        checkpoint=100,
        alpha=0.1,
        horizon=steps,
    )

    num_policies = policies.shape[0]
    horizon = policies.shape[1]
    per_time_sa_occ = np.zeros((num_policies, horizon, num_states, num_actions))
    sa_occ = np.zeros((num_policies, num_states, num_actions))
    true_reward_checkpoints = np.zeros(num_policies)
    proxy_reward_checkpoints = np.zeros(num_policies)

    for power in range(num_policies):
        sa_occ[power], per_time_sa_occ[power] = calculate_occupancy_measure(
            policies[power],
            sparse_transitions,
            num_states,
            num_actions,
            horizon,
        )
        true_reward_checkpoints[power] = (sa_occ[power] * true_rewards).sum()
        proxy_reward_checkpoints[power] = (sa_occ[power] * proxy_rewards).sum()
    plt.plot(np.arange(num_policies), proxy_reward_checkpoints, label="Proxy Reward")
    plt.plot(np.arange(num_policies), true_reward_checkpoints, label="True Reward")
    plt.legend()
    plt.xlabel("Policy Checkpoint")
    plt.ylabel("Reward")
    plt.title("Proxy and True Reward")
    plt.savefig(
        plot_filepath
        + "/reward"
        # + str(use_decay)
        + "_Level"
        + str(level)
        + "_Steps"
        + str(steps)
        + ".png"
    )

    plt.close()

    TV_distances = calculate_occupancy_measure_TV(sa_occ, num_policies)

    dist_plot = plt.matshow(TV_distances)
    plt.colorbar(dist_plot)
    plt.title("State Action Occupancy TV Distances")
    plt.savefig(
        plot_filepath
        + "/TV"
        # + str(use_decay)
        + "_Level"
        + str(level)
        + "_Steps"
        + str(steps)
        + ".png"
    )

    plt.close()

    per_time_KL_distances = calculate_action_dist_KL(per_time_sa_occ, policies)

    dist_plot = plt.matshow(per_time_KL_distances)
    plt.colorbar(dist_plot)
    plt.title("Policy KL Distances")
    plt.savefig(
        plot_filepath
        + "/KL"
        # + str(use_decay)
        + "_Level"
        + str(level)
        + "_Steps"
        + str(steps)
        + ".png"
    )

    plt.close()

    print("Saved plots!")


if __name__ == "__main__":
    diff_steps = [100]  # [50, 100]

    for level in [-1]:  # range(1, 4):
        for steps in diff_steps:
            print("Level: " + str(level))
            print("Steps: " + str(steps))
            generate_plots("data/", level, steps)
            # generate_plots("occupancy_measures/", level, steps, 1)
