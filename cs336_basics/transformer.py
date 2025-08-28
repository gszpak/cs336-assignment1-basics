import einops
import numpy
import torch


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        self._weights = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = numpy.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self._weights, mean=0.0, std=std, a=(-3 * std), b=(3 * std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self._weights, "... d_in, d_out d_in -> ... d_out")
