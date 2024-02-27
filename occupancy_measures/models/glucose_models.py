import logging
from typing import List, Optional, TypedDict, cast

import torch
from gymnasium.spaces import utils
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from torch import nn

from .model_with_discriminator import (
    DISCRIMINATOR_STATE,
    ModelWithDiscriminator,
    ModelWithDiscriminatorConfig,
)

logger = logging.getLogger(__name__)


class GlucoseModelConfig(TypedDict, total=False):
    hidden_size: int
    num_layers: int
    action_scale: float
    use_cgm_for_obs: bool
    use_subcutaneous_glucose_obs: bool


def normalize_obs(obs):
    obs[..., 0] = (obs[..., 0] - 100) / 100
    obs[..., 1] = obs[..., 1] * 10
    return obs


class GlucoseModel(ModelWithDiscriminator):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        ModelWithDiscriminator.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )

        assert model_config["vf_share_layers"] is True
        custom_model_config: ModelWithDiscriminatorConfig = self.model_config[
            "custom_model_config"
        ]
        env_specific_config = cast(
            GlucoseModelConfig, custom_model_config["env_specific_config"]
        )
        self.use_cgm = env_specific_config.get("use_cgm_for_obs", False)
        self.use_subcutaneous_glucose_obs = env_specific_config.get(
            "use_subcutaneous_glucose_obs", False
        )
        self.hidden_size = env_specific_config.get("hidden_size", 64)
        self.num_layers = env_specific_config.get("num_layers", 3)
        self.action_scale = env_specific_config.get("action_scale", 10)
        lstm_input_size = 2
        if self.use_cgm or self.use_subcutaneous_glucose_obs:
            lstm_input_size = 1
            if custom_model_config["discriminator_state_dim"] != 1 and self.use_cgm:
                logger.warn(
                    "Only keeping the first dimension of glucose observations as that contains the CGM reading"
                )
            elif custom_model_config["discriminator_state_dim"] != 1:
                logger.warn(
                    "Only keeping the subcutaneous glucose values as the observation to feed into the discriminator"
                )
            custom_model_config["discriminator_state_dim"] = 1

        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.hidden_size, num_outputs)
        self.value_head = nn.Linear(self.hidden_size, 1)

        # Since we scale the action outputs by the action_scale, initialize the weights
        # and biases of the action head to be small to prevent large initial
        # action outputs.
        self.action_head.weight.data.mul_(1 / self.action_scale)
        self.action_head.bias.data.mul_(1 / self.action_scale)

        # LSTM Discriminator
        self.lstm_discriminator = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )

        self.in_dim = self.hidden_size
        if self.use_action_for_disc:
            self.in_dim += utils.flatdim(action_space)

        self.fc_discriminator = self._build_discriminator(self.in_dim)

        self.discriminator_submodules.append("lstm_discriminator")
        self.discriminator_submodules.append("fc_discriminator")

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict[SampleBatch.OBS].permute(0, 2, 1).clone()
        # Normalize observations
        normalize_obs(obs)
        lstm_out, _ = self.lstm(obs)
        self._backbone_out = lstm_out[:, -1, :]
        # self._backbone_out = self.backbone(obs[:, :, :].mean(axis=1))
        acs = self.action_head(self._backbone_out) * self.action_scale
        return acs, state

    def get_discriminator(self) -> List[nn.Module]:
        return [self.lstm_discriminator, self.fc_discriminator]

    def gradient_penalty(self, given_disc_scores=None):
        if self.last_disc_input is None:
            logger.warn(
                "Gradient penalty cannot be calculated without discriminator scores being calculated first!"
            )
            return 0
        elif given_disc_scores is not None:
            discriminator_scores = given_disc_scores
            return super().gradient_penalty(given_disc_scores=discriminator_scores)
        elif (not self.use_history_for_disc) or self.discriminator_state_dim:
            return super().gradient_penalty()
        else:
            discriminator_scores = self.fc_discriminator(self.last_disc_input)
            return super().gradient_penalty(given_disc_scores=discriminator_scores)

    def discriminator(
        self,
        input_dict,
        seq_lens: Optional[torch.Tensor] = None,
    ):
        # if we don't want to use history for the discriminator, or if we have a separately specified state (other than the CGM only readings), just return the FC net used in ModelWithDiscriminator
        if (not self.use_history_for_disc) or DISCRIMINATOR_STATE in input_dict:
            normalized_input_dict = input_dict
            if self.use_subcutaneous_glucose_obs:
                normalized_input_dict[DISCRIMINATOR_STATE] = normalized_input_dict[
                    DISCRIMINATOR_STATE
                ][
                    :, 12:
                ]  # only use subcutaneous glucose
            else:
                normalized_input_dict = {key: input_dict[key] for key in input_dict}
                normalized_input_dict[SampleBatch.OBS] = (
                    input_dict[SampleBatch.OBS].clone().permute(0, 2, 1)
                )
                normalize_obs(normalized_input_dict[SampleBatch.OBS])
                normalized_input_dict[SampleBatch.OBS] = normalized_input_dict[
                    SampleBatch.OBS
                ].permute(0, 2, 1)
            return super().discriminator(normalized_input_dict, seq_lens)

        # can use less history
        obs = input_dict[SampleBatch.OBS].clone()
        indices = torch.tensor(tuple(range(*self.history_range)))
        obs = obs[:, :, indices].permute(0, 2, 1)
        normalize_obs(obs)
        if self.use_cgm:
            obs = obs[:, :, :1]
        obs_embed, _ = self.lstm_discriminator(obs)
        obs_embed = obs_embed[:, -1, :]
        if self.use_action_for_disc:
            actions = input_dict[SampleBatch.ACTIONS]
            net_input = self._get_concatenated_obs_action(obs_embed, actions)
        else:
            net_input = obs_embed
        self.last_disc_input = net_input
        return self.fc_discriminator(net_input)

    def value_function(self):
        return self.value_head(self._backbone_out).squeeze(-1)


ModelCatalog.register_custom_model("glucose", GlucoseModel)
