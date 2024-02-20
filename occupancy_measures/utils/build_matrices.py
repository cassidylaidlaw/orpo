import argparse
import os
from copy import deepcopy
from typing import Dict

import numpy as np

from occupancy_measures.envs.tomato_environment import (
    Tomato_Environment,
    create_simple_example,
)

path = "./"

boards = []


def build_matrices(
    env: Tomato_Environment,
    visited: Dict,
    transitions: Dict,
    true_rewards: Dict,
    proxy_rewards: Dict,
):
    count = 0
    visited[str(env)] = count
    queue = [env]
    boards.append(env.board_visualization())

    while queue:
        env = queue.pop(0)
        for i in range(len(env.actions)):
            action = env.actions[i]
            env_copy = deepcopy(env)
            env_copy.step(action)
            curr_state = str(env_copy)

            if curr_state not in visited:
                count += 1
                boards.append(env_copy.board_visualization())
                visited[curr_state] = count
                queue.append(env_copy)

            previous_state = str(env)
            state_action_tup = (visited[previous_state], i)
            transitions[state_action_tup] = visited[curr_state]
            true_rewards[state_action_tup] = env_copy.true_reward()
            proxy_rewards[state_action_tup] = env_copy.proxy_reward()


def save_files(filepath: str, level: int, toy=False):
    filename, _ = create_simple_example(filepath, level)
    config = {}
    config["filepath"] = filename
    config["horizon"] = 100
    config["reward_fun"] = "proxy"
    if toy:
        config["use_noop"] = True
        config["dry_distance"] = 0
    env = Tomato_Environment(config)

    saved_matrices_path = filepath + "saved_matrices"
    if not os.path.exists(path + saved_matrices_path):
        os.makedirs(path + saved_matrices_path)

    visited: Dict = {}
    transitions: Dict = {}
    true_rewards: Dict = {}
    proxy_rewards: Dict = {}
    build_matrices(env, visited, transitions, true_rewards, proxy_rewards)
    transitions_mat = np.zeros((len(visited), len(env.actions)))
    true_rewards_mat = np.zeros((len(visited), len(env.actions)))
    state_true_reward_mat = np.zeros((len(visited), len(visited)))
    proxy_rewards_mat = np.zeros((len(visited), len(env.actions)))
    state_proxy_reward_mat = np.zeros((len(visited), len(visited)))
    for state_action_tup, next_state in transitions.items():
        previous_state = state_action_tup[0]
        action = state_action_tup[1]
        true_reward = true_rewards[state_action_tup]
        proxy_reward = proxy_rewards[state_action_tup]
        transitions_mat[previous_state][action] = next_state
        true_rewards_mat[previous_state][action] = true_reward
        state_true_reward_mat[previous_state][next_state] = true_reward
        proxy_rewards_mat[previous_state][action] = proxy_reward
        state_proxy_reward_mat[previous_state][next_state] = proxy_reward

    with open(
        saved_matrices_path + "/raw_transitions_" + str(level) + ".npy", "wb"
    ) as transition_file:
        np.save(transition_file, np.array(transitions))
    with open(
        saved_matrices_path + "/transitions_" + str(level) + ".npy", "wb"
    ) as transition_file:
        np.save(transition_file, transitions_mat)
    with open(
        saved_matrices_path + "/proxy_rewards_" + str(level) + ".npy", "wb"
    ) as proxy_rewards_file:
        np.save(proxy_rewards_file, proxy_rewards_mat)
    with open(
        saved_matrices_path + "/true_rewards_" + str(level) + ".npy", "wb"
    ) as true_rewards_file:
        np.save(true_rewards_file, true_rewards_mat)
    with open(
        saved_matrices_path + "/state_proxy_rewards_" + str(level) + ".npy", "wb"
    ) as proxy_rewards_file:
        np.save(proxy_rewards_file, state_proxy_reward_mat)
    with open(
        saved_matrices_path + "/state_true_rewards_" + str(level) + ".npy", "wb"
    ) as true_rewards_file:
        np.save(true_rewards_file, state_true_reward_mat)
    with open(
        saved_matrices_path + "/boards_" + str(level) + ".npy", "wb"
    ) as boards_file:
        np.save(boards_file, boards)

    print("Saved files for level " + str(level))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath", type=str, help="filepath to save matrices", default="data/"
    )
    parser.add_argument(
        "--level",
        nargs="+",
        type=int,
        help="levels of tomato environment to build",
        default=[1],
    )
    parser.add_argument("--toy", type=bool, help="to build toy example", default=False)
    args = parser.parse_args()
    if args.toy:
        levels = [-1]
    else:
        levels = args.level
    for level in levels:
        save_files(args.filepath, level, args.toy)
