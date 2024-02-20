import logging
from typing import Dict, List, Type, Union, cast

import numpy as np
import torch
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ResultDict, TensorType
from ray.tune.registry import register_trainable

from ..models.reward_model import RewardModel

logger = logging.getLogger(__name__)

TRUE_REWARD = "true_reward"


class LearnedRewardConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a AlgorithmConfig instance."""
        super().__init__(algo_class=algo_class or LearnedRewardAlgorithm)
        self.input = "sampler"
        self.batch_mode = "complete_episodes"
        self.train_batch_size = 2000
        self.sgd_minibatch_size = 128
        self.num_rollout_workers = 0
        self.evaluation_num_workers = 0
        self.gamma = 1.0
        self.noise_prob = 0.0
        self.action_info_key = None
        self.rew_clip = 50.0

        # dummy entry
        self.exploration_config = {
            "type": "StochasticSampling",
        }

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        gamma: float = NotProvided,  # type: ignore[assignment]
        noise_prob: float = NotProvided,  # type: ignore[assignment]
        action_info_key: Union[List, str] = NotProvided,  # type: ignore[assignment]
        rew_clip: float = NotProvided,  # type: ignore[assignment]
        **kwargs,
    ) -> "LearnedRewardConfig":
        """Sets the training related configuration.
        Args:
        Returns:
            gamma: discount factor to calculate sum of rewards along trajectory
                for preferences
            noise_prob: probability of having some noise in the simulated preferences
            action_info_key: Instead of using SampleBatch actions for prediction,
                train to predict rewards using this key from the info dictionaries.
            rew_clip: clipping for rewards to avoid overflow in softmax probability
                calculation
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if gamma is not NotProvided:
            self.gamma = gamma
        if noise_prob is not NotProvided:
            self.noise_prob = noise_prob
        if action_info_key is not NotProvided:
            self.action_info_key = action_info_key
        if rew_clip is not NotProvided:
            self.rew_clip = rew_clip

        return self


class LearnedRewardPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        self._initialize_loss_from_dummy_batch()
        self.rng = np.random.default_rng(seed=self.config["seed"])

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, RewardModel)
        device = train_batch[SampleBatch.OBS].device
        reward_model_loss = torch.tensor(0, dtype=torch.float32, device=device)
        accuracy = torch.tensor(0, dtype=torch.float32, device=device)
        actions = train_batch[SampleBatch.ACTIONS]
        actions_numpy = convert_to_numpy(actions)
        if self.config["action_info_key"]:
            action_info_keys = self.config["action_info_key"]
            if isinstance(action_info_keys, str):
                action_info_keys = [action_info_keys]
            logged_action_info_key_warning = False
            for i in range(len(train_batch)):
                found_action_info_key = False
                for action_info_key in action_info_keys:
                    if action_info_key in train_batch:
                        actions_numpy[i] = train_batch[action_info_key][i].cpu().numpy()
                        found_action_info_key = True
                        break

                    if not found_action_info_key and not logged_action_info_key_warning:
                        logger.warning(
                            f"action_info_key not found in info dict (looked for any "
                            f"of {action_info_keys})"
                        )
                        logged_action_info_key_warning = True

        actions = torch.from_numpy(actions_numpy).to(self.device)

        rewards_sequences = add_time_dimension(
            train_batch[SampleBatch.REWARDS],
            seq_lens=train_batch[SampleBatch.SEQ_LENS],
            framework="torch",
            time_major=False,
        )
        obs_sequences = add_time_dimension(
            train_batch[SampleBatch.OBS],
            seq_lens=train_batch[SampleBatch.SEQ_LENS],
            framework="torch",
            time_major=False,
        )
        acs_sequences = add_time_dimension(
            actions,
            seq_lens=train_batch[SampleBatch.SEQ_LENS],
            framework="torch",
            time_major=False,
        )
        if TRUE_REWARD in train_batch:
            true_reward_sequences = add_time_dimension(
                train_batch[TRUE_REWARD],
                seq_lens=train_batch[SampleBatch.SEQ_LENS],
                framework="torch",
                time_major=False,
            )
        else:
            logger.warn("True reward key not in info dictionary!")
            true_reward_sequences = torch.zeros(rewards_sequences.shape)

        num_sequences = train_batch[SampleBatch.SEQ_LENS].shape[0]
        num_sequences = num_sequences if num_sequences % 2 == 0 else (num_sequences - 1)
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)
        trajectories_iter = iter(indices)
        trajectory_pairs = list(zip(trajectories_iter, trajectories_iter))
        for indices_pair in trajectory_pairs:
            traj1 = self._create_sample_batch(
                rewards_sequences,
                acs_sequences,
                obs_sequences,
                true_reward_sequences,
                indices_pair[0],
            )
            traj2 = self._create_sample_batch(
                rewards_sequences,
                acs_sequences,
                obs_sequences,
                true_reward_sequences,
                indices_pair[1],
            )
            predicted_reward_probs = self._calculate_boltzmann_pred_probs(
                model, traj1, traj2
            ).to(device)
            true_reward_label = self._calculate_true_reward_comparisons(
                traj1, traj2
            ).to(device)
            if (
                predicted_reward_probs > 1
                or predicted_reward_probs < 0
                or torch.isnan(predicted_reward_probs)
            ):
                logger.error(
                    f"Invalid value for predicted rewards! Value: {predicted_reward_probs}"
                )
            elif (
                true_reward_label > 1
                or true_reward_label < 0
                or torch.isnan(true_reward_label)
            ):
                logger.error(
                    f"Invalid value for true reward-based labels! Value: {true_reward_label}"
                )

            predicted_label = (predicted_reward_probs > 0.5).float()
            accuracy += (predicted_label == true_reward_label).float().mean()
            reward_model_loss += torch.nn.functional.binary_cross_entropy(
                predicted_reward_probs, true_reward_label
            )
        model.tower_stats["reward_pred_accuracy"] = accuracy / len(trajectory_pairs)
        model.tower_stats["reward_model_loss"] = reward_model_loss
        return reward_model_loss

    def _create_sample_batch(self, rewards, actions, obs, infos, index):
        return {
            SampleBatch.REWARDS: rewards[index],
            SampleBatch.ACTIONS: actions[index],
            SampleBatch.OBS: obs[index],
            TRUE_REWARD: infos[index],
        }

    """
    Calculates boltzmann probability that one trajectory will be chosen
    over another based on the predicted reward.
    """

    def _calculate_boltzmann_pred_probs(self, model, traj1, traj2):
        traj1_preds = model.learned_reward(traj1).flatten()
        traj2_preds = model.learned_reward(traj2).flatten()
        preds_diff = self._calculate_discounted_sum_and_diffs(traj1_preds, traj2_preds)
        softmax_probs = 1 / (1 + preds_diff.exp())
        probs = (0.5 * self.config["noise_prob"]) + (
            (1 - self.config["noise_prob"]) * softmax_probs
        )
        return probs

    """
    Compares two trajectories based on their true rewards using the calculated
    softmax probabilities
    """

    def _calculate_true_reward_comparisons(self, traj1, traj2):
        traj1_true_rewards = traj1[TRUE_REWARD]
        traj2_true_rewards = traj2[TRUE_REWARD]
        rewards_diff = self._calculate_discounted_sum_and_diffs(
            traj1_true_rewards, traj2_true_rewards
        ).cpu()
        # softmax probability that traj 1 would be chosen over traj 2 based on the true reward
        probs = torch.tensor(1 / (1 + np.exp(rewards_diff.cpu().numpy()))).to(
            torch.float32
        )
        return (torch.rand(probs.size(), device=probs.device) < probs).float()

    def _calculate_discounted_sum_and_diffs(self, traj1_rews, traj2_rews):
        discounts = self.config["gamma"] ** torch.arange(
            len(traj1_rews), device=traj1_rews.device
        )
        rewards_diff = (discounts * (traj2_rews - traj1_rews)).sum(axis=0)
        return torch.clip(
            rewards_diff, -self.config["rew_clip"], self.config["rew_clip"]
        )

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = {
            "reward_model_loss": torch.mean(
                torch.stack(
                    cast(List[torch.Tensor], self.get_tower_stats("reward_model_loss"))
                )
            ),
            "reward_pred_accuracy": torch.mean(
                torch.stack(
                    cast(
                        List[torch.Tensor], self.get_tower_stats("reward_pred_accuracy")
                    )
                )
            ),
        }
        return cast(Dict[str, TensorType], convert_to_numpy(stats))


class LearnedRewardAlgorithm(Algorithm):
    config: LearnedRewardConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return LearnedRewardConfig()

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy]:
        if config.framework_str == "torch":
            return LearnedRewardPolicy
        else:
            raise NotImplementedError("Only PyTorch is supported.")

    def training_step(self) -> ResultDict:
        assert self.workers is not None
        # Collect SampleBatches from sample workers until we have a full batch.
        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config.train_batch_size
        )
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Add state_in_0 to train_batch to make sure that RLlib splits it into
        # contiguous sequences.
        assert isinstance(train_batch, MultiAgentBatch)
        for _, policy_batch in train_batch.policy_batches.items():
            if "state_in_0" not in policy_batch:
                policy_batch["state_in_0"] = np.zeros(len(policy_batch))

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


register_trainable("LearnedRewardAlgorithm", LearnedRewardAlgorithm)
