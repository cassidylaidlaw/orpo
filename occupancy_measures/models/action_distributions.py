import torch
import torch.nn.functional as F
from ray.rllib.models.torch.torch_action_dist import TorchBeta, TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import List, TensorType


class UnclampedBeta(TorchBeta):
    """
    Version of TorchBeta that doesn't clamp the inputs to be in a small range.
    """

    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        low: float = 0.0,
        high: float = 1.0,
    ):
        TorchDistributionWrapper.__init__(self, inputs, model)
        # Stabilize input parameters (possibly coming from a linear layer).
        self.inputs = F.softplus(self.inputs) + 1.0  # type: ignore
        self.low = low
        self.high = high
        alpha, beta = torch.chunk(self.inputs, 2, dim=-1)  # type: ignore
        # Note: concentration0==beta, concentration1=alpha (!)
        self.dist = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
