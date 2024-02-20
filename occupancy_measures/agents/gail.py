import logging
from typing import Dict, List, Type, Union, cast

import torch
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_gae_for_sample_batch
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models import ActionDistribution, ModelV2
from ray.rllib.offline.json_reader import JsonReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID,
    MultiAgentBatch,
    SampleBatch,
    concat_samples,
)
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.sgd import minibatches
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict, TensorType
from ray.tune.registry import register_trainable
from torch.nn import functional as F

from ..models.model_with_discriminator import ModelWithDiscriminator

logger = logging.getLogger(__name__)


class GailPolicy(PPOTorchPolicy):
    demonstration_data: SampleBatch
    current_demonstration_minibatches: List[SampleBatch]

    def __init__(self, observation_space, action_space, config):
        config = GAIL.merge_algorithm_configs(
            {
                **GAIL.get_default_config(),
                "worker_index": None,
            },
            config,
        )

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)
        self._init_demonstrations(
            config["demonstration_input"], config["demonstration_num_episodes"]
        )

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

        self._initialize_loss_from_dummy_batch()

    def _init_demonstrations(
        self,
        demonstration_input: str,
        demonstration_num_episodes: int = 1,
    ):
        logger.info("loading demonstrations...")
        demonstration_reader = JsonReader(demonstration_input)
        self.demonstration_num_episodes = demonstration_num_episodes

        all_demonstration_batches: List[SampleBatch] = []
        batches_fnames: List[str] = demonstration_reader.files
        for batches_fname in batches_fnames:
            with open(batches_fname, "r") as batches_file:
                for line in batches_file:
                    line = line.strip()
                    if line:
                        batch = demonstration_reader._from_json(line)
                        assert isinstance(batch, SampleBatch)
                        all_demonstration_batches.append(batch)
        self.demonstration_data = concat_samples(all_demonstration_batches)
        self.current_demonstration_minibatches = []

    def _get_demonstration_batch(self) -> SampleBatch:
        if len(self.current_demonstration_minibatches) == 0:
            self.current_demonstration_minibatches = list(
                minibatches(
                    self.demonstration_data,
                    sgd_minibatch_size=self.config["sgd_minibatch_size"],
                )
            )

        return cast(
            SampleBatch,
            self._lazy_tensor_dict(self.current_demonstration_minibatches.pop()),
        )

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, ModelWithDiscriminator)

        # Train discriminator.
        discriminator_policy_scores = model.discriminator(train_batch)

        demonstration_batch = self._get_demonstration_batch()
        discriminator_demonstration_scores = model.discriminator(demonstration_batch)

        discriminator_loss = (
            F.softplus(discriminator_demonstration_scores).mean()
            + F.softplus(-discriminator_policy_scores).mean()
        )

        # Store additional stats in policy for stats_fn.
        model.tower_stats["discriminator_loss"] = discriminator_loss
        model.tower_stats["discriminator_policy_score"] = (
            discriminator_policy_scores.mean()
        )
        model.tower_stats["discriminator_demonstration_score"] = (
            discriminator_demonstration_scores.mean()
        )

        demonstration_model_out, _ = model.from_batch(demonstration_batch)
        demonstration_action_dist = dist_class(demonstration_model_out, model)
        model.tower_stats["demonstration_cross_entropy"] = (
            -demonstration_action_dist.logp(
                demonstration_batch[SampleBatch.ACTIONS]
            ).mean()
        )

        model.tower_stats["discriminator_reward"] = train_batch[
            SampleBatch.REWARDS
        ].mean()

        ppo_loss = super().loss(
            model,
            dist_class,
            train_batch,
        )

        loss = ppo_loss + discriminator_loss
        return loss

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().stats_fn(train_batch)
        for stats_key, tower_stats_id in [
            ("discriminator/loss", "discriminator_loss"),
            ("discriminator/policy_score", "discriminator_policy_score"),
            ("discriminator/demonstration_score", "discriminator_demonstration_score"),
            ("discriminator/reward", "discriminator_reward"),
            ("demonstration_cross_entropy", "demonstration_cross_entropy"),
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


class GAIL(PPO):
    @classmethod
    def get_default_config(cls) -> AlgorithmConfigDict:
        return {
            **super().get_default_config(),
            # Directory with offline-format data to use as demonstrations.
            "demonstration_input": None,
            # How many episodes of demonstration data to use per SGD step.
            "demonstration_num_episodes": 1,
        }

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return GailPolicy
        else:
            raise NotImplementedError()

    def augment_reward_with_discriminator(
        self, train_batch: SampleBatch
    ) -> SampleBatch:
        wrapped = False

        if isinstance(train_batch, SampleBatch):
            multiagent_batch = MultiAgentBatch(
                {DEFAULT_POLICY_ID: train_batch}, train_batch.count
            )
            wrapped = True
        else:
            multiagent_batch = train_batch

        for policy_id, batch in multiagent_batch.policy_batches.items():
            policy = self.get_policy(policy_id)
            assert isinstance(policy, GailPolicy)
            batch = multiagent_batch.policy_batches[policy_id]
            model = cast(ModelWithDiscriminator, policy.model)
            batch_for_discriminator = batch.copy().decompress_if_needed()
            discriminator_policy_scores = model.discriminator(
                policy._lazy_tensor_dict(batch_for_discriminator)
            )
            batch[SampleBatch.REWARDS] = (
                F.softplus(-discriminator_policy_scores)[:, 0].cpu().detach().numpy()
            )
            # Need to recompute advantages.
            multiagent_batch.policy_batches[policy_id] = compute_gae_for_sample_batch(
                policy,
                batch,
            )

        if wrapped:
            train_batch = multiagent_batch.policy_batches[DEFAULT_POLICY_ID]
        else:
            train_batch = multiagent_batch

        return train_batch

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
        )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        assert isinstance(train_batch, MultiAgentBatch)
        train_batch = self.augment_reward_with_discriminator(train_batch)

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        train_results: ResultDict
        # Train
        if self.config["simple_optimizer"]:
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

        # For each policy: update KL scale.
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("GAIL", GAIL)
