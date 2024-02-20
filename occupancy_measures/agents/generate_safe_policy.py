import logging
from collections.abc import Iterable
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_trainable

from ..agents.orpo import ORPO, ORPOConfig, ORPOPolicy

logger = logging.getLogger(__name__)

DEFAULT_POLICY_ID = "safe_policy0"
TRUE_REWARDS = "true_reward"


class SafePolicyConfig(ORPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or SafePolicyGenerationAlgorithm)
        self.safe_policy_action_dist_input_info_key: Optional[str] = None
        self.safe_policy_action_log_std = -3.0
        self.categorical_eps = 0.9
        # dummy entry
        self.exploration_config = {
            "type": "StochasticSampling",
        }

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()

        if self.safe_policy_action_dist_input_info_key == "":
            pandemic_keys = [
                "S0",
                "S1",
                "S2",
                "S3",
                "S4",
                "S0-4-0",
                "S0-4-0-FI",
                "S0-4-0-GI",
                "swedish_strategy",
                "italian_strategy",
            ]
            pandemic_keys_str = ", ".join(f"{key}" for key in pandemic_keys)

            raise ValueError(
                "The SafePolicy class should not be used unless using actions specified in an info dictionary."
                "Please specify a key in the info dictionary where the safe policy actions are located."
                'For the traffic environment, the key is "acc_controller_actions";'
                'for the glucose environment, the key is "glucose_pid_controller";'
                "for the pandemic environment, the key can be any combination of the following: {}".format(
                    pandemic_keys_str
                )
            )

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        safe_policy_action_dist_input_info_key: Optional[str] = NotProvided,  # type: ignore[assignment]
        safe_policy_action_log_std: Union[int, float] = NotProvided,  # type: ignore[assignment]
        categorical_eps: Union[int, float] = NotProvided,  # type: ignore[assignment]
        **kwargs,
    ) -> "SafePolicyConfig":
        """Sets the training related configuration.
        Args:
            safe_policy_action_dist_input_info_key: the models when overriding
                the RL actions with some other controller actions from the
                environment, specify which key to look for in info dict
            safe_policy_action_log_std: the standard deviation to use for
                user-specified actions
            categorical_eps: when using actions specified by a categorical
                distribution, specify this variable
        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)

        if safe_policy_action_dist_input_info_key is not NotProvided:
            self.safe_policy_action_dist_input_info_key = (
                safe_policy_action_dist_input_info_key
            )
        if safe_policy_action_log_std is not NotProvided:
            self.safe_policy_action_log_std = safe_policy_action_log_std
        if categorical_eps is not NotProvided:
            self.categorical_eps = categorical_eps

        return self


class SafePolicy(ORPOPolicy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(
            observation_space,
            action_space,
            config,
        )

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        postprocessed_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        # post-process the trajectory such that if controller actions are being
        # used, they will be used to replace the actions provided by the policy
        # network.
        if self.config["safe_policy_action_dist_input_info_key"] is not None:
            safe_policy_action = np.zeros_like(postprocessed_batch[SampleBatch.ACTIONS])
            true_rewards = np.zeros_like(postprocessed_batch[SampleBatch.REWARDS])
            assert self.dist_class is not None
            for t in range(len(postprocessed_batch)):
                if not isinstance(postprocessed_batch[SampleBatch.INFOS][t], Iterable):
                    break
                if (t % self.config["rollout_fragment_length"] == 0) or (
                    t > 0 and postprocessed_batch[SampleBatch.DONES][t - 1]
                ):
                    continue
                action = postprocessed_batch[SampleBatch.INFOS][t][
                    self.config["safe_policy_action_dist_input_info_key"]
                ]
                safe_policy_action[t] = action
                postprocessed_batch[SampleBatch.ACTION_DIST_INPUTS][t] = (
                    self.get_timestep_safe_policy_action_dist_inputs(action)
                )
                postprocessed_batch[SampleBatch.ACTIONS][t] = self.dist_class(
                    postprocessed_batch[SampleBatch.ACTION_DIST_INPUTS][t],
                    None,
                ).sample()
                true_rewards[t] = self._get_true_reward_from_info(
                    postprocessed_batch[SampleBatch.INFOS][t]
                )
            postprocessed_batch[
                self.config["safe_policy_action_dist_input_info_key"]
            ] = safe_policy_action
            postprocessed_batch[TRUE_REWARDS] = true_rewards
        return postprocessed_batch

    def _get_true_reward_from_info(self, timestep_info):
        if TRUE_REWARDS in timestep_info:
            true_reward = timestep_info[TRUE_REWARDS]
        else:
            for key, value in timestep_info.items():
                if key.lower() in TRUE_REWARDS or TRUE_REWARDS in key.lower():
                    true_reward = value
                    break
        return true_reward

    def get_timestep_safe_policy_action_dist_inputs(
        self, action: Union[int, float, np.ndarray]
    ) -> torch.Tensor:
        """
        Given the action distribution class for the environment, use the provided
        actions to generate the inputs for the action distribution.

        Note: For the glucose and traffic environments, we decided to use a beta
        distribution since they have continuous action spaces with a limited
        range of values, and for the pandemic and tomato environments, we decided
        to use a simpler categorical distribution since they have discrete action
        spaces.
        """

        if self.dist_class is not None and issubclass(
            self.dist_class, TorchCategorical
        ):
            assert type(action) is int
            if np.random.uniform(0, 1) <= self.config["categorical_eps"]:
                return torch.Tensor([action])
            else:
                return torch.Tensor([self.action_space.sample()])
        action = np.array(action)
        assert isinstance(action, np.ndarray)
        first_half = action
        second_half = self.config["safe_policy_action_log_std"]
        if self.dist_class is not None and issubclass(self.dist_class, TorchBeta):
            first_half, second_half = self._postprocess_beta_dist_inputs(action)
        action_dist_inputs = np.empty((first_half.shape[0] * 2,))
        action_dist_inputs[: len(action_dist_inputs) // 2] = first_half
        action_dist_inputs[len(action_dist_inputs) // 2 :] = second_half
        if self.dist_class is not None and issubclass(self.dist_class, TorchBeta):
            torch_action_dist_inputs = torch.log(
                torch.exp(torch.from_numpy(action_dist_inputs) - 1) - 1
            )
        else:
            torch_action_dist_inputs = torch.from_numpy(action_dist_inputs)
        if torch.any(torch.isnan(torch_action_dist_inputs)):
            raise ValueError("Action distribution inputs contain nans!")
        return torch_action_dist_inputs

    def _postprocess_beta_dist_inputs(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.dist_class is not None

        action = np.array(action)
        if len(action.shape) == 0:
            action = action[None]

        dist = self.dist_class(np.full(action.shape[0] * 2, 1.1), None)
        low = dist.low
        high = dist.high
        mu = (action - low) / (high - low)
        mu = np.clip(mu, 0.01, 0.99)
        sigma = np.exp(self.config["safe_policy_action_log_std"])
        alpha = mu * (mu - mu**2 - sigma**2) / (sigma**2)
        beta = (1 - mu) * (mu - mu**2 - sigma**2) / (sigma**2)

        # to ensure that the alpha and beta values are greater than 1, to adhere
        # to the RLLib torch Beta distribution definition
        normalizer = (
            np.minimum(np.minimum(alpha, beta), np.full(alpha.shape, 1.1)) / 1.1
        )
        alpha = alpha / normalizer
        beta = beta / normalizer

        return alpha, beta


class SafePolicyGenerationAlgorithm(ORPO):
    config: SafePolicyConfig

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return SafePolicyConfig()

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy]:
        if config.framework_str == "torch":
            return SafePolicy
        else:
            raise NotImplementedError("Only PyTorch is supported.")

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config.train_batch_size
        )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        assert isinstance(train_batch, MultiAgentBatch)

        # Need to split into episodes since postprocess_trajectory expects
        # single-episode batches.
        current_policy = self.get_policy(DEFAULT_POLICY_ID)
        postprocessed_episodes: List[SampleBatch] = []
        current_batch = train_batch.policy_batches[DEFAULT_POLICY_ID]
        for episode_batch in current_batch.split_by_episode():
            postprocessed_episodes.append(
                current_policy.postprocess_trajectory(episode_batch)
            )
        train_batch.policy_batches[DEFAULT_POLICY_ID] = concat_samples(
            postprocessed_episodes
        )

        # Train
        train_results: ResultDict
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)
        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("SafePolicyGenerationAlgorithm", SafePolicyGenerationAlgorithm)
