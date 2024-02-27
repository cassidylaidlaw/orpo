from typing import Callable, List, Optional, TypedDict

import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces import utils
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType
from torch import nn


class RewardModelConfig(TypedDict, total=False):
    reward_model_width: int
    reward_model_depth: int
    normalize_obs: Optional[Callable]


class RewardModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_model_config: RewardModelConfig = model_config.get(
            "custom_model_config", {}
        )
        self.rew_model_width = custom_model_config.get("reward_model_width", 256)
        self.rew_model_depth = custom_model_config.get("reward_model_depth", 2)
        self.normalize_obs = custom_model_config.get("normalize_obs", None)
        if self.normalize_obs is not None:
            assert callable(
                self.normalize_obs
            ), "Must specify a function for normalizing the observations in-place"

        rew_in_dim = utils.flatdim(action_space) + utils.flatdim(obs_space)

        self.fc_reward_net = self._build_reward_model(rew_in_dim)

        self.net = TorchFC(
            self.obs_space,
            self.action_space,
            self.num_outputs,
            model_config,
            name="net",
        )

    def get_initial_state(self) -> List[TensorType]:
        return [torch.zeros(1)]

    def forward(self, input_dict, state=[torch.zeros(1)], seq_lens=None):
        logits, state = self.net.forward(input_dict, state, seq_lens)
        return logits, [s + 1 for s in state]

    def value_function(self):
        return self.net.value_function()

    def learned_reward(self, input_dict):
        obs = input_dict[SampleBatch.OBS].flatten(1)
        if self.normalize_obs is not None:
            self.normalize_obs(obs)
        actions = input_dict[SampleBatch.ACTIONS]
        net_input = self._get_concatenated_obs_action(obs, actions)
        predicted_rew = self.fc_reward_net(net_input)
        return predicted_rew

    def _build_reward_model(
        self,
        in_dim: int,
        out_dim: int = 1,
    ) -> nn.Module:
        rew_fcnet_hiddens = [self.rew_model_width] * self.rew_model_depth
        dims = [in_dim] + rew_fcnet_hiddens + [out_dim]
        layers: List[nn.Module] = []
        for dim1, dim2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(dim1, dim2))
            layers.append(nn.LeakyReLU())
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _get_concatenated_obs_action(self, obs, actions):
        if isinstance(self.action_space, spaces.Discrete):
            num_actions = int(self.action_space.n)
            encoded_actions = F.one_hot(actions.long(), num_actions)
            net_input = torch.cat([obs, encoded_actions], dim=1)
        else:
            net_input = torch.cat([obs, actions], dim=1)
        net_input = net_input.to(torch.float32)
        return net_input


ModelCatalog.register_custom_model("reward_model", RewardModel)
