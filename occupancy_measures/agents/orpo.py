import logging
from ast import Str
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import tree
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models import ActionDistribution, ModelV2
from ray.rllib.models.torch.torch_action_dist import (
    TorchBeta,
    TorchCategorical,
    TorchDiagGaussian,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.policy.torch_mixins import (
    EntropyCoeffSchedule,
    KLCoeffMixin,
    LearningRateSchedule,
    ValueNetworkMixin,
)
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import PolicyID, ResultDict, TensorType
from ray.tune.registry import register_trainable
from torch import nn
from torch.nn import functional as F

from ..models.model_with_discriminator import (
    DISCRIMINATOR_STATE,
    ModelWithDiscriminator,
)

logger = logging.getLogger(__name__)


DISCRIMINATOR_REWARDS = "discriminator_rewards"
UNCLIPPED_DISCRIMINATOR_REWARDS = "unclipped_discriminator_rewards"
IS_SAFE_POLICY = "is_safe_policy"
SAFE_POLICY_ACTION_DIST_INPUTS = "safe_policy_action_dist_inputs"


class ORPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or ORPO)
        self.om_divergence_coeffs: Dict[PolicyID, float] = {}
        self.om_divergence_type: Dict[PolicyID, Str] = {}
        self.action_dist_divergence_coeff: Optional[float] = None
        self.action_dist_divergence_type: str = "kl"
        self.update_safe_policy_freq: Optional[int] = None
        self.safe_policy_ids: List[PolicyID] = []
        self.current_policy_id: PolicyID = "current"
        self.train_discriminator_first = True
        self.discriminator_reward_clip = np.inf
        self.num_extra_repeated_safe_policy_batches = 1
        self.discriminator_state_info_key: Optional[str] = None
        self.discriminator_num_sgd_iter: Optional[int] = None

        # parameters to use when using wasserstein distance
        self.wgan_grad_clip = 0.01
        self.wgan_grad_penalty_weight: Optional[float] = None
        self.wasserstein_distance_subtract_mean_safe_policy_score = False

        # beta features
        self.split_om_kl = False
        self.occupancy_measure_kl_target: Dict[PolicyID, float] = {}
        self.use_squared_kl_adaptive_coefficient = False

    @override(AlgorithmConfig)
    def validate(self) -> None:
        super().validate()

        if self.action_dist_divergence_coeff is not None:
            if len(self.safe_policy_ids) != 1:
                raise ValueError(
                    "Must specify exactly one safe policy when using action_dist_divergence_coeff"
                )

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        om_divergence_coeffs: Dict = NotProvided,  # type: ignore[assignment]
        om_divergence_type: Dict = NotProvided,  # type: ignore[assignment]
        action_dist_divergence_coeff: Optional[float] = NotProvided,  # type: ignore[assignment]
        action_dist_divergence_type: str = NotProvided,  # type: ignore[assignment]
        update_safe_policy_freq: Optional[int] = NotProvided,  # type: ignore[assignment]
        safe_policy_ids: List[str] = NotProvided,  # type: ignore[assignment]
        current_policy_id: str = NotProvided,  # type: ignore[assignment]
        train_discriminator_first: bool = NotProvided,  # type: ignore[assignment]
        discriminator_reward_clip: float = NotProvided,  # type: ignore[assignment]
        num_extra_repeated_safe_policy_batches: int = NotProvided,  # type: ignore[assignment]
        discriminator_state_info_key: Optional[str] = NotProvided,  # type: ignore[assignment]
        discriminator_num_sgd_iter: Optional[int] = NotProvided,  # type: ignore[assignment]
        wgan_grad_clip: Union[int, float] = NotProvided,  # type: ignore[assignment]
        wgan_grad_penalty_weight: Optional[Union[int, float]] = NotProvided,  # type: ignore[assignment]
        wasserstein_distance_subtract_mean_safe_policy_score: Optional[
            bool
        ] = NotProvided,  # type: ignore[assignment]
        split_om_kl: bool = NotProvided,  # type: ignore[assignment]
        occupancy_measure_kl_target: Dict = NotProvided,  # type: ignore[assignment]
        use_squared_kl_adaptive_coefficient: bool = NotProvided,  # type: ignore[assignment]
        **kwargs,
    ) -> "ORPOConfig":
        """Sets the training related configuration.
        Args:
            om_divergence_coeffs: weight for the OM divergence allowed
                from the safe policy
            om_divergence_type: what type of distance measure to use
                (kl, tv, wasserstein, chi2, sqrt_chi2)
            action_dist_divergence_coeff: for action distribution regularization,
                set this coefficient
            action_dist_divergence_type: the type of action distribution
                regularization to use (kl, chi2, sqrt_chi2)
            update_safe_policy_freq: when using multiple safe policies,
                set this option to specify with what frequency the safe policy
                should be updated this could be used for detection of reward
                hacking, or if we want to regularize to earlier checkpoints
                during training
            safe_policy_ids: the policy ids for the safe policies
            current_policy_id: the policy id of the policy to train
            train_discriminator_first: whether or not to train the discriminator
                before using it to calculate rewards for the policy
            discriminator_reward_clip: clipping for discriminator rewards to
                help with instability
            num_extra_repeated_safe_policy_batches: train the discriminator
                with this much more data per iteration
            discriminator_state_info_key: when using a different state to
                train the discriminator (i.e., the true state), specify which
                key to look for in info dict
            discriminator_num_sgd_iter: number of epochs to train the
                discriminator; if None, will use the same number of epochs
                as the policy.
            wgan_grad_clip: gradient clipping for the lipschitz condition
            wgan_grad_penalty_weight: specify weight for gradient penalty
            wasserstein_distance_subtract_mean_safe_policy_score: whether or not
                to subtract the mean safe_policy scores. This allows for the
                wasserstein distance to more clearly be calculated.
            split_om_kl: whether or not to split the OM KL calculation
                into action and state calculations
            occupancy_measure_kl_target: target for dynamically tuning the
                occupancy measure kl
            use_squared_kl_adaptive_coefficient: squared KL adaptive coefficient
        Returns:
            This updated AlgorithmConfig object.
        """
        super().training(**kwargs)
        if om_divergence_coeffs is not NotProvided:
            self.om_divergence_coeffs = om_divergence_coeffs
        if om_divergence_type is not NotProvided:
            self.om_divergence_type = om_divergence_type
        if action_dist_divergence_coeff is not NotProvided:
            self.action_dist_divergence_coeff = action_dist_divergence_coeff
        if action_dist_divergence_type is not NotProvided:
            self.action_dist_divergence_type = action_dist_divergence_type
        if update_safe_policy_freq is not NotProvided:
            self.update_safe_policy_freq = update_safe_policy_freq
        if safe_policy_ids is not NotProvided:
            self.safe_policy_ids = safe_policy_ids
        if current_policy_id is not NotProvided:
            self.current_policy_id = current_policy_id

        self._set_discriminator_params(
            train_discriminator_first,
            discriminator_reward_clip,
            num_extra_repeated_safe_policy_batches,
            discriminator_state_info_key,
            discriminator_num_sgd_iter,
        )
        self._set_wasserstein_params(
            wgan_grad_clip,
            wgan_grad_penalty_weight,
            wasserstein_distance_subtract_mean_safe_policy_score,
        )
        self._set_experimental_params(
            split_om_kl,
            occupancy_measure_kl_target,
            use_squared_kl_adaptive_coefficient,
        )

        return self

    def _set_discriminator_params(
        self,
        train_discriminator_first,
        discriminator_reward_clip,
        num_extra_repeated_safe_policy_batches,
        discriminator_state_info_key,
        discriminator_num_sgd_iter,
    ):
        if train_discriminator_first is not NotProvided:
            self.train_discriminator_first = train_discriminator_first
        if discriminator_reward_clip is not NotProvided:
            self.discriminator_reward_clip = discriminator_reward_clip
        if num_extra_repeated_safe_policy_batches is not NotProvided:
            self.num_extra_repeated_safe_policy_batches = (
                num_extra_repeated_safe_policy_batches
            )
        if discriminator_state_info_key is not NotProvided:
            self.discriminator_state_info_key = discriminator_state_info_key
        if discriminator_num_sgd_iter is not NotProvided:
            self.discriminator_num_sgd_iter = discriminator_num_sgd_iter

    def _set_wasserstein_params(
        self,
        wgan_grad_clip,
        wgan_grad_penalty_weight,
        wasserstein_distance_subtract_mean_safe_policy_score,
    ):
        if wgan_grad_clip is not NotProvided:
            self.wgan_grad_clip = wgan_grad_clip
        if wgan_grad_penalty_weight is not NotProvided:
            self.wgan_grad_penalty_weight = wgan_grad_penalty_weight
        if wasserstein_distance_subtract_mean_safe_policy_score is not NotProvided:
            self.wasserstein_distance_subtract_mean_safe_policy_score = (
                wasserstein_distance_subtract_mean_safe_policy_score
            )

    def _set_experimental_params(
        self,
        split_om_kl,
        occupancy_measure_kl_target,
        use_squared_kl_adaptive_coefficient,
    ):
        if split_om_kl is not NotProvided:
            self.split_om_kl = split_om_kl
        if occupancy_measure_kl_target is not NotProvided:
            self.occupancy_measure_kl_target = occupancy_measure_kl_target
        if use_squared_kl_adaptive_coefficient is not NotProvided:
            self.use_squared_kl_adaptive_coefficient = (
                use_squared_kl_adaptive_coefficient
            )


def _beta_chi2(
    alpha_1: torch.Tensor,
    beta_1: torch.Tensor,
    alpha_2: torch.Tensor,
    beta_2: torch.Tensor,
) -> torch.Tensor:
    log_x = torch.where(
        (2 * alpha_1 - alpha_2 > 0) & (2 * beta_1 - beta_2 > 0),
        2 * torch.lgamma(alpha_1 + beta_1)
        + torch.lgamma(alpha_2)
        + torch.lgamma(beta_2)
        - torch.lgamma(alpha_2 + beta_2)
        - 2 * torch.lgamma(alpha_1)
        - 2 * torch.lgamma(beta_1)
        + torch.lgamma(2 * alpha_1 - alpha_2)
        + torch.lgamma(2 * beta_1 - beta_2)
        - torch.lgamma(2 * alpha_1 - alpha_2 + 2 * beta_1 - beta_2),
        torch.nan,
    )
    return cast(torch.Tensor, torch.special.expm1(log_x.sum(dim=-1)))


def chi2_divergence(
    p: ActionDistribution,
    q: ActionDistribution,
) -> torch.Tensor:
    if isinstance(p, TorchCategorical):
        assert isinstance(q, TorchCategorical)
        x: torch.Tensor = (p.dist.probs - q.dist.probs) ** 2 / q.dist.probs
        x[(q.dist.probs == 0) & (p.dist.probs == 0)] = 0
        return x.sum(-1)
    elif isinstance(p, TorchDiagGaussian):
        assert isinstance(q, TorchDiagGaussian)

        mean_1 = p.dist.loc
        mean_2 = q.dist.loc
        std_1 = p.dist.scale
        std_2 = q.dist.scale

        d = 2 * std_2**2 - std_1**2

        x = std_2**2 / (std_1 * torch.sqrt(d)) * torch.exp((mean_1 - mean_2) ** 2 / d)
        return torch.prod(x, dim=-1) - 1
    elif isinstance(p, TorchBeta):
        assert isinstance(q, TorchBeta)

        alpha_1 = p.dist.concentration1
        beta_1 = p.dist.concentration0
        alpha_2 = q.dist.concentration1
        beta_2 = q.dist.concentration0

        # We calculate the chi-squared divergence twice, but the second time only on
        # the values that came out finite. This ensures that backprop doesn't create
        # nan gradients.
        with torch.no_grad():
            first_chi2 = _beta_chi2(alpha_1, beta_1, alpha_2, beta_2)
            mask = torch.isfinite(first_chi2)
            second_chi2 = torch.zeros(len(first_chi2), device=first_chi2.device)
        second_chi2[mask] = _beta_chi2(
            alpha_1[mask], beta_1[mask], alpha_2[mask], beta_2[mask]
        )

        return torch.where(
            mask,
            second_chi2.clip(min=0),
            torch.inf,
        )
    else:
        raise NotImplementedError(f"Unsupported distribution: {type(p)}")


class ORPOPolicy(PPOTorchPolicy):
    is_safe_policy: bool
    action_dist_divergence_coeff: float

    def __init__(self, observation_space, action_space, config):
        config = ORPO.merge_algorithm_configs(
            {
                **ORPO.get_default_config(),
                "worker_index": None,
            },
            config,
            _allow_unknown_configs=True,
        )
        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        self.policy_id = config.get("__policy_id")

        ValueNetworkMixin.__init__(self, config)
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])
        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        KLCoeffMixin.__init__(self, config)

        # The current KL value (as python float).
        self.kl_coeff = self.config["kl_coeff"]
        # Constant target value.
        self.kl_target = self.config["kl_target"]

        self.is_safe_policy = False

        self.action_dist_divergence_coeff = 0
        self.occupancy_measure_divergence_coeff = 0
        self.gamma_scaling = 1 / (1 - self.config["gamma"])

        self._initialize_loss_from_dummy_batch()

    def split_batch(
        self, train_batch: SampleBatch
    ) -> Tuple[Optional[SampleBatch], Optional[SampleBatch]]:
        """
        Split the given SampleBatch into two based on the value of
        train_batch[IS_SAFE_POLICY], reversing ORPO.transfer_batch.

            Parameters:
                train_batch (SampleBatch): training data where the current policy
                data is concatenated to the safe policy data

            Returns:
                safe_policy_batch, current_batch (Tuple(SampleBatch, SampleBatch)):
                A tuple of separate SampleBatches corresponding to each of the policies
        """
        if IS_SAFE_POLICY in train_batch:
            if (
                train_batch.get(SampleBatch.SEQ_LENS) is not None
                and len(train_batch[SampleBatch.SEQ_LENS]) > 0
            ):
                raise RuntimeError("sequences not supported")

            safe_policy_mask = train_batch[IS_SAFE_POLICY].cpu().numpy()
            current_mask = ~safe_policy_mask

            safe_policy_batch: Optional[SampleBatch]
            if np.any(safe_policy_mask):
                safe_policy_batch = SampleBatch(
                    tree.map_structure(
                        lambda value: value[safe_policy_mask], train_batch
                    ),
                    _is_training=train_batch.is_training,
                    _time_major=train_batch.time_major,
                )
                assert safe_policy_batch is not None
                safe_policy_batch.set_get_interceptor(train_batch.get_interceptor)
            else:
                safe_policy_batch = None

            current_batch: Optional[SampleBatch]
            if np.any(current_mask):
                current_batch = SampleBatch(
                    tree.map_structure(lambda value: value[current_mask], train_batch),
                    _is_training=train_batch.is_training,
                    _time_major=train_batch.time_major,
                )
                assert current_batch is not None
                current_batch.set_get_interceptor(train_batch.get_interceptor)
            else:
                current_batch = None

            return safe_policy_batch, current_batch
        else:
            logger.warn(f"'{IS_SAFE_POLICY}' key not found in train batch")
            return train_batch, train_batch

    def _tv_discriminator_loss(
        self,
        current_batch: Optional[SampleBatch],
        safe_policy_batch: Optional[SampleBatch],
        model=None,
    ):
        discriminator_loss: TensorType = 0
        if current_batch is not None:
            discriminator_curr_policy_scores = model.discriminator(current_batch)
            model.tower_stats["discriminator_curr_policy_score"] = (
                discriminator_curr_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss
                + F.tanh(-discriminator_curr_policy_scores).mean()
                # Small penalty to avoid huge discriminator scores.
                + 0.01 * (discriminator_curr_policy_scores**2).mean()
            )
        if safe_policy_batch is not None:
            discriminator_safe_policy_scores = model.discriminator(safe_policy_batch)
            model.tower_stats["discriminator_safe_policy_score"] = (
                discriminator_safe_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss
                + F.tanh(discriminator_safe_policy_scores).mean()
                # Small penalty to avoid huge discriminator scores.
                + 0.01 * (discriminator_safe_policy_scores**2).mean()
            )
        return discriminator_loss

    def _wasserstein_discriminator_loss(
        self,
        current_batch: Optional[SampleBatch],
        safe_policy_batch: Optional[SampleBatch],
        model,
    ):
        discriminator_loss: TensorType = 0
        if current_batch is not None:
            discriminator_curr_policy_scores = model.discriminator(current_batch)
            model.tower_stats["discriminator_curr_policy_score"] = (
                discriminator_curr_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss - discriminator_curr_policy_scores.mean()
            )
            if self.config["wgan_grad_penalty_weight"]:
                discriminator_loss = discriminator_loss + self.config[
                    "wgan_grad_penalty_weight"
                ] * model.gradient_penalty(discriminator_curr_policy_scores)
        if safe_policy_batch is not None:
            discriminator_safe_policy_scores = model.discriminator(safe_policy_batch)
            model.tower_stats["discriminator_safe_policy_score"] = (
                discriminator_safe_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss + discriminator_safe_policy_scores.mean()
            )
            if self.config["wgan_grad_penalty_weight"]:
                discriminator_loss = discriminator_loss + self.config[
                    "wgan_grad_penalty_weight"
                ] * model.gradient_penalty(discriminator_safe_policy_scores)
        return discriminator_loss

    def _kl_discriminator_loss(
        self,
        current_batch: Optional[SampleBatch],
        safe_policy_batch: Optional[SampleBatch],
        model=None,
    ):
        discriminator_loss: TensorType = 0
        if current_batch is not None:
            discriminator_curr_policy_scores = model.discriminator(current_batch)
            model.tower_stats["discriminator_curr_policy_score"] = (
                discriminator_curr_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss
                + F.softplus(-discriminator_curr_policy_scores).mean()
            )
        if safe_policy_batch is not None:
            discriminator_safe_policy_scores = model.discriminator(safe_policy_batch)
            model.tower_stats["discriminator_safe_policy_score"] = (
                discriminator_safe_policy_scores.mean()
            )

            discriminator_loss = (
                discriminator_loss + F.softplus(discriminator_safe_policy_scores).mean()
            )
        return discriminator_loss

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, ModelWithDiscriminator)
        safe_policy_batch, current_batch = self.split_batch(train_batch)
        discriminator_loss = 0
        if self.is_safe_policy:
            # Train discriminator.
            if self.config["om_divergence_type"][self.policy_id] == "tv":
                discriminator_loss = self._tv_discriminator_loss(
                    current_batch,
                    safe_policy_batch,
                    model,
                )
            elif self.config["om_divergence_type"][self.policy_id] == "wasserstein":
                discriminator_loss = self._wasserstein_discriminator_loss(
                    current_batch,
                    safe_policy_batch,
                    model,
                )
            elif (
                self.config["om_divergence_type"][self.policy_id]
                in ["kl", "chi2", "sqrt_chi2"]
                or self.config["om_divergence_type"][self.policy_id]
                == "safe_policy_confidence"
            ):
                discriminator_loss = self._kl_discriminator_loss(
                    current_batch,
                    safe_policy_batch,
                    model,
                )
            else:
                om_divergence_type = self.config["om_divergence_type"][self.policy_id]
                logger.error(f"Unsupported OM divergence: '{om_divergence_type}")
            model.tower_stats["discriminator_loss"] = discriminator_loss
            model.tower_stats["occupancy_measure_divergence_coeff"] = torch.tensor(
                self.occupancy_measure_divergence_coeff,
                device=self.device,
                dtype=torch.float32,
            )

        if DISCRIMINATOR_REWARDS in train_batch:
            model.tower_stats["discriminator_reward"] = train_batch[
                DISCRIMINATOR_REWARDS
            ].mean()
        elif not self.is_safe_policy:
            logger.warn(f"'{DISCRIMINATOR_REWARDS}' key not found in train batch")

        if current_batch is not None and not self.is_safe_policy:
            ppo_loss = super().loss(
                model,
                dist_class,
                current_batch,
            )
            if SAFE_POLICY_ACTION_DIST_INPUTS in current_batch:
                logits, _ = model.__call__(current_batch)
                current_action_dist = dist_class(logits, model)
                safe_policy_action_dist = dist_class(
                    current_batch[SAFE_POLICY_ACTION_DIST_INPUTS],
                    None,
                )

                action_dist_kl = current_action_dist.kl(safe_policy_action_dist)
                model.tower_stats["curr_base_action_distribution_kl"] = torch.mean(
                    action_dist_kl
                ).detach()

                action_dist_chi2 = chi2_divergence(
                    current_action_dist, safe_policy_action_dist
                )
                model.tower_stats["curr_base_action_distribution_chi2_finite_prop"] = (
                    torch.isfinite(action_dist_chi2).float().mean().detach()
                )
                # Not sure of the best way to handle this, but we don't want to ignore
                # states where the chi-squared divergence is infinite, so maybe fine
                # to instead optimize the KL divergence in that case.
                action_dist_chi2 = torch.where(
                    torch.isfinite(action_dist_chi2),
                    action_dist_chi2,
                    action_dist_kl,
                )
                model.tower_stats["curr_base_action_distribution_chi2"] = torch.mean(
                    action_dist_chi2
                ).detach()

                if self.config["action_dist_divergence_type"] == "kl":
                    action_dist_divergence = action_dist_kl
                elif self.config["action_dist_divergence_type"] == "chi2":
                    action_dist_divergence = action_dist_chi2
                elif self.config["action_dist_divergence_type"] == "sqrt_chi2":
                    action_dist_divergence = torch.sqrt(
                        action_dist_chi2.clamp(min=1e-4)
                    )

                if self.action_dist_divergence_coeff != 0:
                    ppo_loss = ppo_loss + (
                        self.gamma_scaling
                        * self.action_dist_divergence_coeff
                        * torch.mean(action_dist_divergence)
                    )

                model.tower_stats["action_dist_divergence_coeff"] = torch.tensor(
                    self.action_dist_divergence_coeff,
                    device=self.device,
                    dtype=torch.float32,
                )
        else:
            ppo_loss = 0

        loss = ppo_loss + discriminator_loss
        return loss

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        stats = super().stats_fn(train_batch)
        for stats_key, tower_stats_id in [
            ("discriminator/loss", "discriminator_loss"),
            ("discriminator/curr_policy_score", "discriminator_curr_policy_score"),
            ("discriminator/safe_policy_score", "discriminator_safe_policy_score"),
            ("discriminator/reward", "discriminator_reward"),
            ("curr_base_action_distribution_kl", "curr_base_action_distribution_kl"),
            (
                "curr_base_action_distribution_chi2",
                "curr_base_action_distribution_chi2",
            ),
            (
                "curr_base_action_distribution_chi2_finite_prop",
                "curr_base_action_distribution_chi2_finite_prop",
            ),
            ("action_dist_divergence_coeff", "action_dist_divergence_coeff"),
            (
                "occupancy_measure_divergence_coeff",
                "occupancy_measure_divergence_coeff",
            ),
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


class ORPO(PPO):
    _allow_unknown_subkeys = PPO._allow_unknown_subkeys + [
        "occupancy_measure_kl_target",
        "om_divergence_coeffs",
        "om_divergence_type",
        "__policy_id",
    ]
    config: ORPOConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls) -> AlgorithmConfig:
        return ORPOConfig()

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfig) -> Type[Policy]:
        if config.framework_str == "torch":
            return ORPOPolicy
        else:
            raise NotImplementedError("Only PyTorch is supported.")

    def setup(self, config):
        super().setup(config)
        if self.config.action_dist_divergence_coeff is not None:
            assert len(self.config.safe_policy_ids) == 1
            self.get_policy(
                self.config.current_policy_id
            ).action_dist_divergence_coeff = self.config.action_dist_divergence_coeff

        for policy_id in self.config.safe_policy_ids:
            policy = self.get_policy(policy_id)
            policy.is_safe_policy = True
            policy.occupancy_measure_divergence_coeff = (
                self.config.om_divergence_coeffs[policy_id]
            )

            def set_safe_policy(worker: RolloutWorker, policy_id=policy_id):
                policy = worker.get_policy(policy_id)
                assert isinstance(policy, ORPOPolicy)
                policy.is_safe_policy = True

            assert self.workers is not None
            self.workers.foreach_worker(set_safe_policy)
            if self.evaluation_workers is not None:
                self.evaluation_workers.foreach_worker(set_safe_policy)

    def transfer_batch_for_discriminator_training(
        self,
        train_batch: MultiAgentBatch,
        current_batch_to_transfer: Optional[SampleBatch] = None,
    ):
        """
        Since we want to train the discriminator that is associated with the
        safe policy models on data from both the safe policy and current
        policy, we need to concatenate the current_batch_to_transfer
        to the data of the safe policy.

        Parameters:
            train_batch (SampleBatch): training data
            current_batch_to_transfer (Optional SampleBatch): specified if
            we would like to pass in data from the current policy that isn't
            from the current training iteration.
        """
        if self.config.current_policy_id in train_batch.policy_batches:
            current_batch = train_batch.policy_batches[self.config.current_policy_id]
            current_batch[IS_SAFE_POLICY] = np.zeros(len(current_batch), dtype=bool)
            current_batch = self._get_discriminator_batch(current_batch)

            if current_batch_to_transfer is None:
                current_batch_to_transfer = current_batch
            elif IS_SAFE_POLICY not in current_batch_to_transfer:
                current_batch_to_transfer[IS_SAFE_POLICY] = np.zeros(
                    len(current_batch_to_transfer), dtype=bool
                )
                current_batch_to_transfer = self._get_discriminator_batch(
                    current_batch_to_transfer
                )

            for policy_id in self.config.safe_policy_ids:
                if policy_id in train_batch.policy_batches:
                    demo_batch = train_batch.policy_batches[policy_id]
                    demo_batch[IS_SAFE_POLICY] = np.ones(len(demo_batch), dtype=bool)
                    demo_batch = self._get_discriminator_batch(demo_batch)
                    train_batch.policy_batches[policy_id] = concat_samples(
                        [current_batch_to_transfer, demo_batch]
                    )

        return train_batch

    def _wasserstein_discriminator_rewards(
        self,
        info,
        model,
        discriminator_policy_scores,
        train_batch,
        safe_policy_id,
        safe_policy,
    ):
        policy_batch = train_batch.policy_batches[safe_policy_id]
        rewards = discriminator_policy_scores[:, 0].cpu().detach().numpy()
        if self.config.wasserstein_distance_subtract_mean_safe_policy_score:
            discriminator_safe_policy_scores = torch.zeros(
                (len(policy_batch), 1), device=safe_policy.device
            )
            for i in range(0, len(policy_batch), self.config.sgd_minibatch_size):
                discriminator_safe_policy_scores[
                    i : i + self.config.sgd_minibatch_size
                ] += model.discriminator(
                    safe_policy._lazy_tensor_dict(
                        policy_batch[i : i + self.config.sgd_minibatch_size].copy()
                    )
                ).detach()
            mean_discriminator_safe_policy_score = torch.mean(
                discriminator_safe_policy_scores
            ).item()
            rewards -= mean_discriminator_safe_policy_score

        info[safe_policy_id]["occupancy_measure_wasserstein"] = np.mean(rewards)
        return rewards

    def _tv_discriminator_rewards(
        self, info, discriminator_policy_scores, safe_policy_id
    ):
        rewards = discriminator_policy_scores.sign().cpu().detach().numpy().flatten()
        info[safe_policy_id]["occupancy_measure_tv"] = float(np.mean(rewards))
        return rewards

    def _safe_policy_confidence_rewards(
        self, info, safe_policy_id, safe_policy, safe_policy_action_dist_inputs
    ):
        safe_policy_action_dist = safe_policy.dist_class(
            torch.from_numpy(safe_policy_action_dist_inputs).cpu().detach(),
            safe_policy.model,
        )
        entropy = safe_policy_action_dist.entropy().cpu().detach().numpy()
        info[safe_policy_id]["safe_policy_confidence"] = entropy.mean()
        return entropy

    def _kl_discriminator_rewards(
        self, info, safe_policy_id, discriminator_policy_scores
    ):
        rewards = discriminator_policy_scores[:, 0].cpu().detach().numpy()
        info[safe_policy_id]["occupancy_measure_kl"] = np.mean(rewards)
        return rewards

    def _chi2_discriminator_rewards(
        self, info, safe_policy_id, discriminator_policy_scores
    ):
        rewards = (discriminator_policy_scores[:, 0].detach().exp() - 1).cpu().numpy()
        info[safe_policy_id]["occupancy_measure_chi2"] = np.mean(rewards)
        return rewards

    def _sqrt_chi2_discriminator_rewards(
        self, info, safe_policy_id, discriminator_policy_scores
    ):
        chi2_rewards = self._chi2_discriminator_rewards(
            info, safe_policy_id, discriminator_policy_scores
        )
        chi2_rewards_sorted = np.sort(chi2_rewards)
        i = int(0.01 * len(chi2_rewards_sorted))
        occupancy_measure_chi2_trimmed = np.mean(chi2_rewards_sorted[i:-i])
        info[safe_policy_id][
            "occupancy_measure_chi2_trimmed"
        ] = occupancy_measure_chi2_trimmed
        if occupancy_measure_chi2_trimmed <= 0:
            rewards = chi2_rewards
        else:
            rewards = chi2_rewards / np.sqrt(occupancy_measure_chi2_trimmed)
        return rewards

    def augment_reward_with_discriminator_and_transfer_action_dist_inputs(
        self, train_batch: MultiAgentBatch
    ) -> Tuple[MultiAgentBatch, Dict[str, Dict[str, float]]]:
        """
        This function takes care of calculating the occupancy measure
        divergence based on what is specified in the config. The divergence
        is used to determine the discriminator rewards, and these discriminator
        rewards are then added to the rewards received by the agent. Finally,
        the current policy's batch is post-processed, so that actions coming
        from safe policies will be appropriately set in future steps.

        Parameters:
            train_batch (MultiAgentBatch): training data from all policies

        Returns:
            train_batch: the processed and augmented train batch
            info: the logged occupancy measure divergence and action
            distribution divergence values, along with other calculated
            metrics useful for logging
        """

        info: Dict[str, Dict[str, float]] = defaultdict(dict)
        if self.config.current_policy_id not in train_batch.policy_batches:
            return train_batch, info

        current_policy = self.get_policy(self.config.current_policy_id)
        current_batch = train_batch.policy_batches[self.config.current_policy_id]
        per_policy_discriminator_reward = np.zeros(len(current_batch))
        for policy_id in self.config.safe_policy_ids:
            if policy_id in train_batch.policy_batches:
                policy = self.get_policy(policy_id)
                occupancy_measure_divergence_coeff = (
                    policy.occupancy_measure_divergence_coeff
                )
                if not isinstance(policy, ORPOPolicy):
                    logger.warn("A non-ORPO policy is being used for regularization!")
                    continue
                model = cast(ModelWithDiscriminator, policy.model)

                safe_policy_action_dist_inputs = self.get_safe_policy_dist_inputs(
                    current_batch=current_batch,
                    safe_policy_model=model,
                    safe_policy=policy,
                )
                current_batch[SAFE_POLICY_ACTION_DIST_INPUTS] = (
                    safe_policy_action_dist_inputs
                )
                current_action_dist_inputs = (
                    torch.from_numpy(current_batch[SampleBatch.ACTION_DIST_INPUTS])
                    .cpu()
                    .detach()
                )
                assert (
                    policy.dist_class is not None
                    and current_policy.dist_class is not None
                )
                safe_policy_action_dist = policy.dist_class(
                    torch.from_numpy(safe_policy_action_dist_inputs).cpu().detach(),
                    policy.model,
                )
                current_action_dist = current_policy.dist_class(
                    current_action_dist_inputs,
                    current_policy.model,
                )
                action_distribution_kl = current_action_dist.kl(safe_policy_action_dist)
                info[policy_id]["action_distribution_kl"] = torch.mean(
                    action_distribution_kl
                ).item()

                discriminator_policy_scores = torch.zeros(
                    (len(current_batch), 1), device=policy.device
                )
                for i in range(0, len(current_batch), self.config.sgd_minibatch_size):
                    discriminator_policy_scores[
                        i : i + self.config.sgd_minibatch_size
                    ] += model.discriminator(
                        policy._lazy_tensor_dict(
                            current_batch[i : i + self.config.sgd_minibatch_size].copy()
                        )
                    ).detach()
                if self.config.om_divergence_type[policy_id] == "wasserstein":
                    distance = self._wasserstein_discriminator_rewards(
                        info,
                        model,
                        discriminator_policy_scores,
                        train_batch,
                        policy_id,
                        policy,
                    )
                elif self.config.om_divergence_type[policy_id] == "tv":
                    distance = self._tv_discriminator_rewards(
                        info, discriminator_policy_scores, policy_id
                    )
                elif (
                    self.config.om_divergence_type[policy_id]
                    == "safe_policy_confidence"
                ):
                    distance = self._safe_policy_confidence_rewards(
                        info, policy_id, policy, safe_policy_action_dist_inputs
                    )
                elif self.config.om_divergence_type[policy_id] == "kl":
                    distance = self._kl_discriminator_rewards(
                        info, policy_id, discriminator_policy_scores
                    )
                elif self.config.om_divergence_type[policy_id] == "chi2":
                    distance = self._chi2_discriminator_rewards(
                        info, policy_id, discriminator_policy_scores
                    )
                elif self.config.om_divergence_type[policy_id] == "sqrt_chi2":
                    distance = self._sqrt_chi2_discriminator_rewards(
                        info, policy_id, discriminator_policy_scores
                    )
                else:
                    logger.error(
                        "No other OM-based divergences are currently supported."
                    )
                if occupancy_measure_divergence_coeff != 0:
                    distance[~np.isfinite(distance)] = 0
                    per_policy_discriminator_reward += (
                        -occupancy_measure_divergence_coeff * distance
                    )

        assert np.all(np.isfinite(per_policy_discriminator_reward))
        current_batch[UNCLIPPED_DISCRIMINATOR_REWARDS] = per_policy_discriminator_reward
        per_policy_discriminator_reward = np.clip(
            per_policy_discriminator_reward,
            -self.config.discriminator_reward_clip,
            self.config.discriminator_reward_clip,
        )
        current_batch[DISCRIMINATOR_REWARDS] = per_policy_discriminator_reward
        current_batch[SampleBatch.REWARDS] = (
            current_batch[SampleBatch.REWARDS] + current_batch[DISCRIMINATOR_REWARDS]
        )
        assert np.all(np.isfinite(current_batch[SampleBatch.REWARDS]))
        episode_batches = current_batch.split_by_episode()
        episode_rewards_with_discriminator = []
        unclipped_episode_rewards_with_discriminator = []
        for i in range(len(episode_batches)):
            episode_rewards_with_discriminator.append(
                np.sum(episode_batches[i][SampleBatch.REWARDS])
            )
            unclipped_episode_rewards_with_discriminator.append(
                np.sum(
                    episode_batches[i][UNCLIPPED_DISCRIMINATOR_REWARDS]
                    - episode_batches[i][DISCRIMINATOR_REWARDS]
                    + episode_batches[i][SampleBatch.REWARDS]
                )
            )

        info[self.config.current_policy_id][
            "episode_reward_with_discriminator_mean"
        ] = np.mean(episode_rewards_with_discriminator)
        info[self.config.current_policy_id][
            "episode_reward_with_unclipped_discriminator_mean"
        ] = np.mean(unclipped_episode_rewards_with_discriminator)

        # Need to split into episodes since postprocess_trajectory expects
        # single-episode batches.
        postprocessed_episodes: List[SampleBatch] = []
        for episode_batch in current_batch.split_by_episode():
            postprocessed_episodes.append(
                current_policy.postprocess_trajectory(episode_batch)
            )
        train_batch.policy_batches[self.config.current_policy_id] = concat_samples(
            postprocessed_episodes
        )
        return train_batch, info

    def get_safe_policy_dist_inputs(
        self,
        current_batch: SampleBatch,
        safe_policy_model: ModelV2,
        safe_policy: ORPOPolicy,
    ):
        """
        The model from the safe policy is used to generate the safe policy's
        action distribution inputs.

        Parameters:
            current_batch (SampleBatch): data from the current policy
            which will have its actions replaced and action distribution
            inputs added
            safe_policy_model (ModelV2): used to calculate the action
            distribution inputs if we don't want to replace actions
            safe_policy (ORPOPolicy): used in the event that the actions
            aren't replaced
        """
        assert isinstance(safe_policy_model, nn.Module)
        safe_policy_action_dist_inputs = np.empty_like(
            current_batch[SampleBatch.ACTION_DIST_INPUTS]
        )
        for batch_offset in range(
            0, len(current_batch), self.config.sgd_minibatch_size
        ):
            minibatch_action_dist_inputs, _ = safe_policy_model.__call__(
                safe_policy._lazy_tensor_dict(
                    current_batch[
                        batch_offset : batch_offset + self.config.sgd_minibatch_size
                    ].copy()
                )
            )
            safe_policy_action_dist_inputs[
                batch_offset : batch_offset + self.config.sgd_minibatch_size
            ] = (minibatch_action_dist_inputs.cpu().detach().numpy())

        return safe_policy_action_dist_inputs

    def _get_discriminator_batch(self, data_batch: SampleBatch) -> SampleBatch:
        # this function is used if we would like to pass in specific information
        # into the discriminator that has been provided in the "info" of the
        # inputted SampleBatch (such as the true state of the environment)
        if self.config.discriminator_state_info_key is not None:
            assert SampleBatch.INFOS in data_batch
            data_batch[DISCRIMINATOR_STATE] = np.array(
                [
                    data_batch[SampleBatch.INFOS][t][
                        self.config.discriminator_state_info_key
                    ]
                    for t in range(len(data_batch))
                ]
            )
        return data_batch

    def _wgan_parameter_clipping(self, train_batch):
        # if wasserstein OM divergence is being used, this takes care of
        # gradient clipping for the Lipschitz condition.
        if self.config.wgan_grad_penalty_weight is None:
            for policy_id in self.config.safe_policy_ids:
                if (
                    policy_id in train_batch.policy_batches
                    and self.config.om_divergence_type[policy_id] == "wasserstein"
                ):
                    policy = self.get_policy(policy_id)
                    model = cast(ModelWithDiscriminator, policy.model)
                    for discriminator in model.get_discriminator():
                        for param in discriminator.parameters():
                            param.data.clamp_(
                                -self.config.wgan_grad_clip,
                                self.config.wgan_grad_clip,
                            )

    def update_occupancy_measure_divergence_coeffs(self, total_kl, policy_id):
        # This function implements the adaptive KL setup (probably going to be removed)
        policy = self.get_policy(policy_id)
        divergence_coeff = policy.occupancy_measure_divergence_coeff
        if self.config.use_squared_kl_adaptive_coefficient:
            divergence_coeff = self.config.om_divergence_coeffs[policy_id] * total_kl
        elif total_kl > 2.0 * self.config.occupancy_measure_kl_target[policy_id]:
            divergence_coeff *= 1.5
        elif total_kl < 0.5 * self.config.occupancy_measure_kl_target[policy_id]:
            divergence_coeff *= 0.5
        policy.occupancy_measure_divergence_coeff = divergence_coeff

        if (
            self.config.action_dist_divergence_coeff
            and self.config.use_squared_kl_adaptive_coefficient
        ):
            self.get_policy(
                self.config.current_policy_id
            ).action_dist_divergence_coeff = (
                self.config.action_dist_divergence_coeff * total_kl
            )
        elif self.config.split_om_kl:
            self.get_policy(
                self.config.current_policy_id
            ).action_dist_divergence_coeff = divergence_coeff

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config.train_batch_size
        )

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        assert isinstance(train_batch, MultiAgentBatch)

        train_batch = self.transfer_batch_for_discriminator_training(train_batch)
        for policy_id in self.config.safe_policy_ids:
            if policy_id in train_batch.policy_batches:
                policy_batch = train_batch.policy_batches[policy_id]
                train_batch.policy_batches[policy_id] = concat_samples(
                    [
                        policy_batch
                        for _ in range(
                            self.config.num_extra_repeated_safe_policy_batches
                        )
                    ]
                )

        if not self.config.train_discriminator_first:
            (
                train_batch,
                occupancy_measure_info,
            ) = self.augment_reward_with_discriminator_and_transfer_action_dist_inputs(
                train_batch
            )

        # only train the discriminator using data from the current and safe
        # policies after batch transfer
        train_results_safe_policy: ResultDict
        safe_policy_batches = {}
        for policy_id in self.config.safe_policy_ids:
            if policy_id in train_batch.policy_batches:
                safe_policy_batches[policy_id] = train_batch.policy_batches[policy_id]
        safe_policy_train_batch = MultiAgentBatch(
            policy_batches=safe_policy_batches, env_steps=train_batch.env_steps()
        )
        num_sgd_iter = self.config.num_sgd_iter
        if self.config.discriminator_num_sgd_iter is not None:
            self.config._is_frozen = False
            self.config.num_sgd_iter = self.config.discriminator_num_sgd_iter
            self.config._is_frozen = True
        if self.config.simple_optimizer:
            train_results_safe_policy = train_one_step(self, safe_policy_train_batch)
        else:
            train_results_safe_policy = multi_gpu_train_one_step(
                self, safe_policy_train_batch
            )
        self.config._is_frozen = False
        self.config.num_sgd_iter = num_sgd_iter
        self.config._is_frozen = True

        if self.config.train_discriminator_first:
            (
                train_batch,
                occupancy_measure_info,
            ) = self.augment_reward_with_discriminator_and_transfer_action_dist_inputs(
                train_batch
            )

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])

        # Train policy
        train_results: ResultDict
        current_policy_batch = {}
        if self.config.current_policy_id in train_batch.policy_batches:
            current_policy_batch[self.config.current_policy_id] = (
                train_batch.policy_batches[self.config.current_policy_id]
            )
            current_policy_train_batch = MultiAgentBatch(
                policy_batches=current_policy_batch, env_steps=train_batch.env_steps()
            )
            if self.config.simple_optimizer:
                train_results = train_one_step(self, current_policy_train_batch)
            else:
                train_results = multi_gpu_train_one_step(
                    self, current_policy_train_batch
                )

        train_results.update(train_results_safe_policy)

        # parameter clipping for Lipschitz constraint
        self._wgan_parameter_clipping(train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # remove old policies and shift policies down using their weights
        if (
            self.config.update_safe_policy_freq is not None
            and self.iteration % self.config.update_safe_policy_freq == 0
        ):
            policy_ids = self.config.safe_policy_ids + [self.config.current_policy_id]
            for i in range(len(policy_ids) - 1):
                assert self.config.policies_to_train is not None
                prev_policy_id = self.config.policies_to_train[i]
                policy_id = self.config.policies_to_train[i + 1]
                weights = (
                    self.workers.local_worker().get_policy(policy_id).get_weights()
                )
                self.workers.local_worker().get_policy(prev_policy_id).set_weights(
                    weights
                )

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

        # Add info about occupancy measure divergence.
        for policy_id, policy_occupancy_measure_info in occupancy_measure_info.items():
            train_results[policy_id].update(policy_occupancy_measure_info)
            if (
                self.config.occupancy_measure_kl_target
                or self.config.use_squared_kl_adaptive_coefficient
            ) and policy_id in self.config.safe_policy_ids:
                total_kl = policy_occupancy_measure_info.get("occupancy_measure_kl", 0)
                if self.config.split_om_kl or self.config.action_dist_divergence_coeff:
                    action_distribution_kl = policy_occupancy_measure_info.get(
                        "action_distribution_kl", 0
                    )
                    total_kl += action_distribution_kl
                    train_results[policy_id][
                        "total_om_action_distribution_kl"
                    ] = total_kl
                total_kl = np.clip(total_kl, 0, None)
                self.update_occupancy_measure_divergence_coeffs(total_kl, policy_id)

        return train_results


register_trainable("ORPO", ORPO)
