from typing import Any, Optional, cast

import torch
from gymnasium.wrappers import FlattenObservation
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import MultiAgentDict, MultiEnvDict
from ray.tune.registry import ENV_CREATOR, _global_registry, register_env
from typing_extensions import TypedDict

from ..agents.learned_reward_algorithm import LearnedRewardAlgorithm
from ..models.reward_model import RewardModel
from ..utils.training_utils import load_algorithm


class LearnedRewardWrapperConfig(TypedDict):
    env: str
    """Name of the environment to wrap."""

    env_config: dict
    """Configuration to be passed to the wrapped environment."""

    reward_fn_checkpoint: str
    """Checkpoint to reward net that will be used to replace the proxy reward"""


class LearnedRewardWrapper(MultiAgentEnv):
    """
    Environment wrapper that replaces the proxy reward with the reward
    predicted by the provided reward model
    """

    base_env: MultiAgentEnv
    config: LearnedRewardWrapperConfig

    def __init__(self, config: LearnedRewardWrapperConfig):
        base_env_name = config["env"]
        base_env_config = config["env_config"]
        env_creator = _global_registry.get(ENV_CREATOR, base_env_name)
        self.base_env_config = config["env_config"]
        base_env = env_creator(base_env_config)
        if isinstance(base_env, MultiAgentEnv):
            self.base_env = base_env
        else:
            env_creator = make_multi_agent(env_creator)
            self.base_env = env_creator(base_env_config)

        self.config = config

        reward_algorithm = load_algorithm(
            self.config["reward_fn_checkpoint"],
            LearnedRewardAlgorithm,
            config_updates={
                "num_rollout_workers": 0,
                "num_gpus": 0,
                "num_gpus_per_worker": 0,
                "env": self.config["env"],
                "env_config": self.config["env_config"],
                "evaluation_num_workers": 0,
            },
        )
        if reward_algorithm.workers is not None:
            (policy_id,) = (
                reward_algorithm.workers.local_worker().get_policies_to_train()
            )
        else:
            policy_id = "safe_policy0"

        self.reward_model = cast(
            RewardModel, reward_algorithm.get_policy(policy_id).model
        )
        self.reward_fn = self.reward_model.learned_reward
        self.action_space = (
            self.base_env.action_space
        )  # dict.fromkeys(self.get_agent_ids(), self.base_env.action_space)
        self.observation_space = (
            self.base_env.observation_space
        )  # dict.fromkeys(self.get_agent_ids(), self.base_env.observation_space)
        self.flattened_obs_space = FlattenObservation(self.base_env)

        self.prev_obs = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    def reset(self, *, seed=None, options=None):
        reset_result, _ = self.base_env.reset()
        self.prev_obs = self.flattened_obs_space.observation(reset_result[0])
        return reset_result, {}

    def observation_space_contains(self, x: MultiAgentDict) -> bool:
        return all(self.observation_space.contains(x[agent]) for agent in x)

    def action_space_contains(self, x: MultiAgentDict) -> bool:
        return all(self.action_space.contains(x[agent]) for agent in x)

    def action_space_sample(self, agent_ids: Optional[list] = None) -> MultiAgentDict:
        if agent_ids is None:
            agent_ids = self.get_agent_ids()
        return {
            agent_id: self.action_space.sample()
            for agent_id in agent_ids
            if agent_id != "__all__"
        }

    def observation_space_sample(
        self, agent_ids: Optional[list] = None
    ) -> MultiEnvDict:
        if agent_ids is None:
            agent_ids = list(range(len(self.envs)))
        obs = {agent_id: self.observation_space.sample() for agent_id in agent_ids}

        return obs

    def step(self, action_dict):
        base_obs = {}
        base_reward = {}
        base_terminated = {}
        base_truncated = {}
        base_infos = {}
        if bool(action_dict):
            (
                base_obs,
                base_reward,
                base_terminated,
                base_truncated,
                base_infos,
            ) = self.base_env.step(action_dict)
            if self.prev_obs is not None:
                for id, _ in action_dict.items():
                    prev_obs_tensor = torch.tensor(self.prev_obs)[None, :]
                    action_tensor = torch.tensor(action_dict[id])[None]
                    input_dict = {
                        SampleBatch.OBS: prev_obs_tensor,
                        SampleBatch.ACTIONS: action_tensor,
                    }
                    reward = self.reward_fn(input_dict)
                    if (
                        "reward_scale" in self.base_env_config
                        and self.base_env_config["reward_scale"] is not None
                    ):
                        reward *= self.base_env_config["reward_scale"]
                    base_reward = {id: reward.item()}
                    for info_key in base_infos[id].keys():
                        if "proxy" in info_key:
                            base_infos[id][info_key] = reward.item()
                            break

                    self.prev_obs = self.flattened_obs_space.observation(base_obs[id])
        return (
            base_obs,
            base_reward,
            base_terminated,
            base_truncated,
            base_infos,
        )


register_env("learned_reward_wrapper", lambda config: LearnedRewardWrapper(config))
