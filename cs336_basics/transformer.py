import math
import einops
import numpy
import torch
from jaxtyping import Float, Int

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weights = _init_linear(out_features, in_features, device=device, dtype=dtype)
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


class SwigluFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff or math.floor(8 * d_model / 3)
        if d_ff % 64 != 0:
            raise ValueError("Internal dimension of the FFN should be a multiple of 64")
        self.w1 = _init_linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = _init_linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = _init_linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(
            self,
            x: Float[torch.Tensor, "... d_model"]
        ) -> Float[torch.Tensor, "... d_model"]:
        w1_x = einops.einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        w3_x = einops.einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        return einops.einsum(
            self.w2, (w1_x * torch.sigmoid(w1_x)) * w3_x,
            "d_model d_ff, ... d_ff -> ... d_model"
        )


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        sequence_positions = torch.arange(max_seq_len, device=device)
        emb_positions = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)
        )
        angles = einops.einsum(sequence_positions, emb_positions, "i, j -> i j")
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotations = einops.rearrange(
            torch.stack([cos, -sin, sin, cos], dim=-1),
            "... (r c) -> ... r c", r=2, c=2
        )
        self.register_buffer("rotations", rotations, persistent=False)

    def forward(
            self,
            x: Float[torch.Tensor, "... seq_len d_k"],
            token_positions: Int[torch.Tensor, "... seq_len"]
        ) -> Float[torch.Tensor, "... seq_len d_k"]:
        seq_len = token_positions.shape[-1]
        rotations = self.get_buffer("rotations")[:seq_len]
        seq_rotations = rotations[token_positions]
        pairwise_grouped = einops.rearrange(x, "... seq_len (d two) -> ... seq_len d two", two=2)
        rotated = einops.einsum(
            pairwise_grouped, seq_rotations,
            "... seq_len d two, seq_len d two1 two -> ... seq_len d two1")
        return einops.rearrange(rotated, "... seq_len d two -> ... seq_len (d two)")


def _init_linear(d1: int, d2: int, device=None, dtype=None) -> torch.nn.Parameter:
    weights = torch.zeros((d1, d2), device=device, dtype=dtype)
    std = numpy.sqrt(2 / (d1 + d2))
    torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=(-3 * std), b=(3 * std))
    return torch.nn.Parameter(data=weights)
