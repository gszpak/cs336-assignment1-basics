import einops
import numpy
import torch
from jaxtyping import Float

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weights = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = numpy.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=(-3 * std), b=(3 * std))
        self.weights = torch.nn.Parameter(data=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embeddings = torch.zeros((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embeddings, mean=0.0, std=1, a=-3, b=3)
        self.embeddings = embeddings

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        gains = torch.ones(d_model, device=device, dtype=dtype)
        self.gains = torch.nn.Parameter(data=gains)
        self.d_model = d_model

    def forward(
            self,
            x: Float[torch.Tensor, "batch_size sequence_length d_model"]
        ) -> Float[torch.Tensor, "batch_size sequence_length d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            (einops.reduce(x ** 2, '... d_model -> ...', 'sum') + torch.full(x.shape[:2], self.eps))
            /
            self.d_model
        )
        result = (x * self.gains) / einops.rearrange(rms, '... -> ... 1')
        return result.to(in_dtype)
