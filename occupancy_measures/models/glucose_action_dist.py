from typing import cast

import torch
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import List

from .action_distributions import UnclampedBeta


class GlucoseBeta(UnclampedBeta):
    def __init__(
        self,
        inputs: List[torch.Tensor],
        model: TorchModelV2,
        low: float = 0,
        high: float = 0.1,
    ):
        # can get low and high from the action space
        # action_space = model.action_space
        # assert(isinstance(action_space, Box))

        self.low = 0
        self.high = 0.1

        super().__init__(inputs, model, self.low, self.high)

    def logp(self, x: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, super().logp(x.clamp(self.low + 1e-3, self.high - 1e-3))
        )

    @override(TorchDistributionWrapper)
    def entropy(self) -> torch.Tensor:
        return cast(torch.Tensor, super().entropy().sum(-1))

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> torch.Tensor:
        return cast(torch.Tensor, super().kl(other).sum(-1))


ModelCatalog.register_custom_action_dist("GlucoseBeta", GlucoseBeta)
