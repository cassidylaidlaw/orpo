import logging
import random
from typing import Dict, List, Optional, Set, Type, Union, cast

import torch
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import ResultDict, TensorType
from ray.tune.registry import register_trainable

logger = logging.getLogger(__name__)


class BCConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        """Initializes a AlgorithmConfig instance."""
        super().__init__(algo_class=algo_class or BC)
        self.input = "sampler"
        self.grad_clip = None
        self.batch_mode = "complete_episodes"
        self.lr = 1e-4
        self.train_batch_size = 2000
        self.sgd_minibatch_size = 100
        self.num_rollout_workers = 0
        self.validation_prop: float = 0
        self.entropy_coeff: float = 0
        self.action_info_key: Optional[Union[List[str], str]] = None

        # dummy entry
        self.exploration_config = {
            "type": "StochasticSampling",
        }

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        validation_prop: float = NotProvided,  # type: ignore[assignment]
        entropy_coeff: float = NotProvided,  # type: ignore[assignment]
        action_info_key: Union[List, str] = NotProvided,  # type: ignore[assignment]
        **kwargs,
    ) -> "BCConfig":
        """Sets the training related configuration.
        Args:
            validation_prop: Use this proportion of the episodes for validation.
            entropy_coeff: for adding an entropy bonus
            action_info_key: name of the key in the info dictionary that contains
            the safe policy actions, should match the keys specified as "safe_policy_action_dist_input_info_key"
            in the SafePolicyConfig
        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if validation_prop is not NotProvided:
            self.validation_prop = validation_prop
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if action_info_key is not NotProvided:
            self.action_info_key = action_info_key

        return self


class BCTorchPolicy(TorchPolicyV2):
    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, TorchModelV2)

        episode_ids: Set[int] = set(train_batch[SampleBatch.EPS_ID].tolist())
        episode_in_validation: Dict[int, bool] = {
            episode_id: random.Random(episode_id).random()
            < self.config["validation_prop"]
            for episode_id in episode_ids
        }
        validation_mask = torch.tensor(
            [
                episode_in_validation[episode_id.item()]
                for episode_id in train_batch[SampleBatch.EPS_ID]
            ],
            dtype=torch.bool,
            device=self.device,
        )

        actions: torch.Tensor = train_batch[SampleBatch.ACTIONS]
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

        model_out, _ = model(train_batch)
        action_dist: ActionDistribution = dist_class(model_out, model)
        logprobs = action_dist.logp(actions)

        bc_loss = -torch.mean(logprobs[~validation_mask])
        model.tower_stats["bc_loss"] = bc_loss
        model.tower_stats["accuracy"] = (
            (action_dist.deterministic_sample() == actions)[~validation_mask]
            .float()
            .mean()
        )

        entropy = action_dist.entropy().mean()
        model.tower_stats["entropy"] = entropy
        loss = bc_loss - self.config["entropy_coeff"] * entropy

        validation_cross_entropy: Optional[torch.Tensor]
        if torch.any(validation_mask):
            validation_cross_entropy = -logprobs[validation_mask].mean()
            model.tower_stats["validation_cross_entropy"] = validation_cross_entropy
        else:
            validation_cross_entropy = None
            model.tower_stats["validation_cross_entropy"] = torch.zeros(size=(0,))

        return loss

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().stats_fn(train_batch)
        for stats_key, tower_stats_id in [
            ("bc_loss", "bc_loss"),
            ("entropy", "entropy"),
            ("accuracy", "accuracy"),
            ("validation/cross_entropy", "validation_cross_entropy"),
        ]:
            try:
                stats[stats_key] = torch.mean(
                    torch.stack(
                        cast(
                            List[torch.Tensor],
                            self.get_tower_stats(tower_stats_id),
                        )
                    )
                )
            except AssertionError:
                pass

        return cast(Dict[str, TensorType], convert_to_numpy(stats))


class BC(Algorithm):
    config: BCConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return BCConfig()

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy]:
        if config.framework_str == "torch":
            return BCTorchPolicy
        else:
            raise NotImplementedError("Only PyTorch is supported.")

    def training_step(self) -> ResultDict:
        assert self.workers is not None
        # Collect SampleBatches from sample workers until we have a full batch.
        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config.train_batch_size
        )
        # infos key must be removed since it can't be converted to a tensor
        train_batch.__delitem__("infos")
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

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


register_trainable("BC", BC)
