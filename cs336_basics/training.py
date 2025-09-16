import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    normalized = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    exp = torch.exp(normalized)
    sum = torch.sum(exp, dim=-1)
    losses = torch.log(sum) - normalized[torch.arange(len(targets)), targets]
    return torch.mean(losses)
