import logging
from typing import Any, List, Mapping, Optional, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces import utils
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from torch import nn

DISCRIMINATOR_STATE = "discriminator_state"

logger = logging.getLogger(__name__)


class ModelWithDiscriminatorConfig(TypedDict, total=False):
    discriminator_width: int
    discriminator_depth: int
    discriminator_state_dim: (
        int  # need to specify this for history to properly be filtered
    )
    use_action_for_disc: bool
    use_history_for_disc: bool
    time_dim: int  # the dimension along which the history is stored.
    history_range: tuple  # range of values along which the history is stored
    env_specific_config: Mapping


EPS = 1e-10


class ModelWithDiscriminator(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        if model_config["use_lstm"]:
            obs_shape = self.obs_space.shape
            assert obs_shape is not None
            self.num_outputs = int(np.prod(obs_shape))

        self.net = TorchFC(
            self.obs_space,
            self.action_space,
            self.num_outputs,
            model_config,
            name="net",
        )

        custom_model_config: ModelWithDiscriminatorConfig = self.model_config.get(
            "custom_model_config", {}
        )
        self.disc_width = custom_model_config.get("discriminator_width", 256)
        self.disc_depth = custom_model_config.get("discriminator_depth", 2)
        self.use_action_for_disc = custom_model_config.get("use_action_for_disc", True)
        self.use_history_for_disc = custom_model_config.get(
            "use_history_for_disc", True
        )
        self.time_dim = custom_model_config.get("time_dim", 1)
        self.history_range = custom_model_config.get("history_range", (-1, 0))
        self.disc_state_dim = custom_model_config.get(
            "discriminator_state_dim", 0
        )  # if the history param is specified, this corresponds to a separately specified discriminator state (like for the true state); otherwise, this corresponds to the number of features per time unit.

        # TODO: incorporate dones / next obs too?
        disc_in_dim = (
            self.disc_state_dim if self.disc_state_dim else utils.flatdim(obs_space)
        )
        if self.use_action_for_disc:
            disc_in_dim += utils.flatdim(action_space)
        self.discriminator_net = self._build_discriminator(disc_in_dim)
        self.last_disc_input = None

        self.discriminator_submodules = ["discriminator_net"]

    def _build_discriminator(
        self,
        in_dim: int,
        out_dim: int = 1,
    ) -> nn.Module:
        disc_fcnet_hiddens = [self.disc_width] * self.disc_depth
        dims = [in_dim] + disc_fcnet_hiddens + [out_dim]
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

    def forward(self, input_dict, state, seq_lens):
        return self.net.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.net.value_function()

    def gradient_penalty(self, given_disc_scores=None):
        if self.last_disc_input is None:
            logger.warn(
                "Gradient penalty cannot be calculated without discriminator scores being calculated first!"
            )
            return 0
        elif given_disc_scores is not None:
            discriminator_scores = given_disc_scores
        else:
            discriminator_scores = self.discriminator_net(self.last_disc_input)
        device = discriminator_scores.device
        gradients = torch.autograd.grad(
            outputs=discriminator_scores,
            inputs=self.last_disc_input,
            grad_outputs=torch.ones(discriminator_scores.size(), device=device),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.flatten(1)
        grad_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + EPS)
        return ((grad_norm - 1) ** 2).mean()

    def get_discriminator(self) -> List[nn.Module]:
        return [self.discriminator_net]

    def process_obs(self, input_dict):
        if DISCRIMINATOR_STATE in input_dict:
            assert (
                self.disc_state_dim
                and input_dict[DISCRIMINATOR_STATE].shape[1] == self.disc_state_dim
            )
            obs = torch.tensor(input_dict[DISCRIMINATOR_STATE])
        elif self.disc_state_dim:
            obs = input_dict[SampleBatch.OBS].clone()
            indices = torch.tensor(tuple(range(*self.history_range)))
            permutation = np.arange(len(obs.shape)).tolist()
            permutation.append(permutation.pop(self.time_dim))
            obs = obs.permute(permutation)[..., indices]
        else:
            obs = input_dict[SampleBatch.OBS]
        obs = obs.flatten(1)
        if self.disc_state_dim and DISCRIMINATOR_STATE not in input_dict:
            obs = obs[:, : self.disc_state_dim]

        return obs

    def discriminator(
        self,
        input_dict,
        seq_lens: Optional[torch.Tensor] = None,
    ):
        obs = self.process_obs(input_dict)

        if self.use_action_for_disc:
            actions = input_dict[SampleBatch.ACTIONS]
            net_input = self._get_concatenated_obs_action(obs, actions)
        else:
            net_input = obs
        self.last_disc_input = net_input
        self.last_disc_input.requires_grad = True
        return self.discriminator_net(net_input)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        *args,
        **kwargs,
    ):
        try:
            return super().load_state_dict(state_dict, strict, *args, **kwargs)
        except RuntimeError as exception:
            exception_str = str(exception)
            if "size mismatch" in exception_str and any(
                discriminator_submodule in exception_str
                for discriminator_submodule in self.discriminator_submodules
            ):
                # This can happen if we load from a safe policy that was trained with
                # a different discriminator architecture. However, since we don't care
                # about the discriminator in the safe policy checkpoint, we can just
                # ignore this error and load the rest of the module, raising a warning.
                logger.warning(
                    "discriminator weights do not match the current discriminator "
                    "architecture; ignoring discriminator weights"
                )
                state_dict_without_discriminator = {
                    key: value
                    for key, value in state_dict.items()
                    if not any(
                        key.split(".")[0] == discriminator_submodule
                        for discriminator_submodule in self.discriminator_submodules
                    )
                }
                return super().load_state_dict(
                    state_dict_without_discriminator, strict=False
                )
            else:
                raise exception


ModelCatalog.register_custom_model("model_with_discriminator", ModelWithDiscriminator)
