import collections
import json
import os
import pickle
from datetime import datetime
from typing import Any, Callable, Container, Dict, Optional, Type, Union, cast

import gymnasium as gym
import ray
import tree
from packaging import version
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.checkpoints import get_checkpoint_info, try_import_msgpack
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.serialization import (
    NOT_SERIALIZABLE,
    gym_space_to_dict,
    serialize_type,
)
from ray.rllib.utils.typing import (
    AlgorithmConfigDict,
    PolicyID,
    PolicyState,
    SampleBatchType,
)
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls

# from .unified_logger_callback import UnifedLoggerCallback

CHECKPOINT_VERSION = version.Version("1.1")


def build_logger_creator(log_dir: str, experiment_name: str):
    experiment_dir = os.path.join(
        log_dir,
        experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in
        <log_dir>/<experiment_name>_<timestamp>
        """

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        return UnifiedLogger(config, experiment_dir)
        # return UnifedLoggerCallback()

    return custom_logger_creator


def deep_transform(u: dict, func) -> dict:
    d: dict = {}
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_transform(d.get(k, {}), v)
        else:
            d[k] = func(v)
    return d


def serialize_dict(config: Dict):
    # Serialize all found spaces to their serialized structs.
    # Note: We cannot use `tree.map_structure` here b/c it doesn't play
    # nice with gym Dicts or Tuples (crashes).

    config = deep_transform(
        config, lambda s: gym_space_to_dict(s) if isinstance(s, gym.Space) else s
    )
    # Serialize all found classes (types) to their classpaths:
    config = tree.map_structure(
        lambda s: serialize_type(s) if isinstance(s, type) else s, config
    )
    # List'ify sets.
    config = tree.map_structure(lambda s: list(s) if isinstance(s, set) else s, config)
    # List'ify `policies`, iff a tuple (these types are not JSON'able).
    ma_config = config.get("multiagent")
    if ma_config is not None:
        if isinstance(ma_config.get("policies"), tuple):
            ma_config["policies"] = list(ma_config["policies"])
    # However, if these "multiagent" settings have been provided directly
    # on the top-level (as they should), we override the settings under
    # "multiagent". Note that the "multiagent" key should no longer be used anyways.
    if isinstance(config.get("policies"), tuple):
        config["policies"] = list(config["policies"])

    # Non-JSON'able items will be masked out via NOT_SERIALIZABLE.
    def _serialize_item(item):
        try:
            json.dumps(item)
        except TypeError:
            return NOT_SERIALIZABLE
        return item

    config = tree.map_structure(_serialize_item, config)

    return config


def export_policy_checkpoint(
    export_dir: str,
    policy_state: PolicyState,
    checkpoint_format: str = "cloudpickle",
) -> None:
    if checkpoint_format not in ["cloudpickle", "msgpack"]:
        raise ValueError(
            f"Value of `checkpoint_format` ({checkpoint_format}) must either be "
            "'cloudpickle' or 'msgpack'!"
        )

    policy_state = cast(dict, policy_state)

    # Write main policy state file.
    os.makedirs(export_dir, exist_ok=True)
    if checkpoint_format == "cloudpickle":
        policy_state["checkpoint_version"] = CHECKPOINT_VERSION
        state_file = "policy_state.pkl"
        with open(os.path.join(export_dir, state_file), "w+b") as f:
            pickle.dump(policy_state, f)
    else:
        msgpack = try_import_msgpack(error=True)
        policy_state["checkpoint_version"] = str(CHECKPOINT_VERSION)
        # Serialize the config for msgpack dumping.
        policy_state["policy_spec"]["config"] = serialize_dict(
            policy_state["policy_spec"]["config"]
        )
        # the actual code for the policy class - converting to an immutable dict
        # doesn't allow for serialization
        policy_state["policy_spec"]["policy_class"] = NOT_SERIALIZABLE
        state_file = "policy_state.msgpck"
        with open(os.path.join(export_dir, state_file), "w+b") as f:
            msgpack.dump(policy_state, f)

    # Write RLlib checkpoint json.
    with open(os.path.join(export_dir, "rllib_checkpoint.json"), "w") as f:
        json.dump(
            {
                "type": "Policy",
                "checkpoint_version": str(policy_state["checkpoint_version"]),
                "format": checkpoint_format,
                "state_file": state_file,
                "ray_version": ray.__version__,
                "ray_commit": ray.__commit__,
            },
            f,
        )


def convert_to_msgpack_checkpoint(
    checkpoint: str,
    msgpack_checkpoint_dir: str,
    run: Union[str, Type[Algorithm]],
) -> str:
    # Try to import msgpack and msgpack_numpy.
    msgpack = try_import_msgpack(error=True)

    # Restore the Algorithm using the python version dependent checkpoint.
    algo = load_algorithm(checkpoint, run=run)
    state = algo.__getstate__()

    # Convert all code in state into serializable data.
    # Serialize the algorithm class.
    state["algorithm_class"] = serialize_type(state["algorithm_class"])
    # Serialize (as much as possible) the algorithm's config object. However, this field
    # will NOT be used when recovering from the msgpack checkpoint as it's impossible to
    # properly serialize things like lambdas, constructed classes, and other
    # code-containing items. Therefore, the user must bring their own original config
    # code when recovering from a msgpack checkpoint.
    state["config"] = serialize_dict(state["config"])

    # Extract policy states from worker state (Policies get their own
    # checkpoint sub-dirs).
    policy_states = {}
    if "worker" in state and "policy_states" in state["worker"]:
        policy_states = state["worker"].pop("policy_states", {})

    # Policy mapping fn.
    state["worker"]["policy_mapping_fn"] = NOT_SERIALIZABLE
    # Is Policy to train function.
    state["worker"]["is_policy_to_train"] = NOT_SERIALIZABLE
    # Filters applied to observation space - we don't use this in our implementation
    for id, filter in state["worker"]["filters"].items():
        state["worker"]["filters"][id] = NOT_SERIALIZABLE

    # Add RLlib checkpoint version (as string).
    state["checkpoint_version"] = str(CHECKPOINT_VERSION)

    # Write state (w/o policies) to disk.
    state_file = os.path.join(msgpack_checkpoint_dir, "algorithm_state.msgpck")
    with open(state_file, "wb") as f:
        msgpack.dump(state, f)

    # Write rllib_checkpoint.json.
    with open(os.path.join(msgpack_checkpoint_dir, "rllib_checkpoint.json"), "w") as f:
        json.dump(
            {
                "type": "Algorithm",
                "checkpoint_version": state["checkpoint_version"],
                "format": "msgpack",
                "state_file": state_file,
                "policy_ids": list(policy_states.keys()),
                "ray_version": ray.__version__,
                "ray_commit": ray.__commit__,
            },
            f,
        )

    # Write individual policies to disk, each in their own sub-directory.
    for pid, policy_state in policy_states.items():
        # From here on, disallow policyIDs that would not work as directory names.
        validate_policy_id(pid, error=True)
        policy_dir = os.path.join(msgpack_checkpoint_dir, "policies", pid)
        os.makedirs(policy_dir, exist_ok=True)
        policy_state["checkpoint_version"] = str(CHECKPOINT_VERSION)
        export_policy_checkpoint(
            policy_dir,
            policy_state=policy_state,
            checkpoint_format="msgpack",
        )

    # Release all resources used by the Algorithm.
    algo.stop()

    return msgpack_checkpoint_dir


def load_policies_from_checkpoint(
    checkpoint_path: str,
    algorithm: Algorithm,
    policy_ids: Optional[Container[PolicyID]] = None,
    policy_mapping_fn: Optional[Callable] = None,
    policies_to_train: Optional[
        Union[
            Container[PolicyID],
            Callable[[PolicyID, Optional[SampleBatchType]], bool],
        ]
    ] = None,
):
    """
    Load policy model weights from a checkpoint and copy them into the given
    algorithm.
    """

    checkpoint_info = get_checkpoint_info(checkpoint_path)

    if checkpoint_info["checkpoint_version"] == version.Version("0.1"):
        raise ValueError(
            "Cannot restore a v0 checkpoint using this method!"
            "In this case, do the following:\n"
            "1) Create a new Algorithm object using your original config.\n"
            "2) Call the `restore()` method of this algo object passing it"
            " your checkpoint dir or AIR Checkpoint object."
        )
    elif checkpoint_info["checkpoint_version"] < version.Version("1.0"):
        raise ValueError(
            "`checkpoint_info['checkpoint_version']` in this method"
            "()` must be 1.0 or later! You are using a checkpoint with "
            f"version v{checkpoint_info['checkpoint_version']}."
        )

    # This is a msgpack checkpoint.
    if checkpoint_info["format"] == "msgpack":
        # User did not provide unserializable function with this call
        # (`policy_mapping_fn`). Note that if `policies_to_train` is None, it
        # defaults to training all policies (so it's ok to not provide this here).
        if policy_mapping_fn is None:
            # Only DEFAULT_POLICY_ID present in this algorithm, provide default
            # implementations of these two functions.
            if checkpoint_info["policy_ids"] == {DEFAULT_POLICY_ID}:
                policy_mapping_fn = AlgorithmConfig.DEFAULT_POLICY_MAPPING_FN
            # Provide meaningful error message.
            else:
                raise ValueError(
                    "You are trying to restore a multi-agent algorithm from a "
                    "`msgpack` formatted checkpoint, which do NOT store the "
                    "`policy_mapping_fn` or `policies_to_train` "
                    "functions! Make sure that when using the "
                    "`Algorithm.from_checkpoint()` utility, you also pass the "
                    "args: `policy_mapping_fn` and `policies_to_train` with your "
                    "call. You might leave `policies_to_train=None` in case "
                    "you would like to train all policies anyway."
                )

    state = Algorithm._checkpoint_info_to_algorithm_state(
        checkpoint_info=checkpoint_info,
        policy_ids=policy_ids,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=policies_to_train,
    )
    workers: WorkerSet = cast(Any, algorithm).workers

    if policy_mapping_fn is None:
        policy_mapping_fn = lambda policy_id: policy_id

    policy_weights = {
        policy_mapping_fn(policy_id): policy_state["weights"]
        for policy_id, policy_state in state["worker"]["policy_states"].items()
    }

    policy_ids = set(
        workers.local_worker().foreach_policy(lambda policy, policy_id: policy_id)
    )
    for policy_id in policy_weights:
        if policy_id not in policy_ids:
            raise ValueError(f"Policy ID '{policy_id}' not found in algorithm.")

    def copy_policy_weights(policy: Policy, policy_id: PolicyID):
        if policy_id in policy_weights:
            policy.set_weights(policy_weights[policy_id])

    workers.foreach_policy(copy_policy_weights)


def load_algorithm_config(checkpoint_path: str) -> AlgorithmConfigDict:
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    return cast(AlgorithmConfigDict, state["config"])


def load_algorithm(
    checkpoint_path: str,
    run: Union[str, Type[Algorithm]],
    config_updates: dict = {},
) -> Algorithm:
    checkpoint_info = get_checkpoint_info(checkpoint_path)
    state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
    config_updates.setdefault("num_rollout_workers", 0)
    state["config"] = Algorithm.merge_algorithm_configs(
        state["config"], config_updates, _allow_unknown_configs=True
    )
    # Create the Trainer from config.
    if isinstance(run, str):
        cls = cast(Type[Algorithm], get_trainable_cls(run))
    else:
        cls = run
    algorithm: Algorithm = cls(
        config=state["config"],
        logger_creator=build_logger_creator("/tmp/ray_results", cls.__name__),
    )
    # Don't load policy_mapping_fn from checkpoint in case we wanted to override it
    # with config updates.
    del state["worker"]["policy_mapping_fn"]
    # Load state from checkpoint.
    algorithm.__setstate__(state)

    return algorithm


def convert_sacred_dict(d):
    d = dict(d)
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = convert_sacred_dict(v)
    return d
