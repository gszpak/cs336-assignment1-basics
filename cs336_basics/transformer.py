import einops
import numpy
import torch


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
