import math
import einops
import numpy
import torch
from jaxtyping import Float, Int, Bool


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weights = torch.zeros((out_features, in_features), device=device, dtype=dtype)
        std = numpy.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=(-3 * std), b=(3 * std))
        self.weight = torch.nn.Parameter(data=weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embeddings = torch.zeros((num_embeddings, embedding_dim), device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(embeddings, mean=0.0, std=1, a=-3, b=3)
        self.weight = torch.nn.Parameter(data=embeddings)

    def forward(
            self, token_ids: Int[torch.Tensor, " ..."]
        ) -> Float[torch.Tensor, "... d_model"]:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float=1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = torch.nn.Parameter(
            data=torch.ones(d_model, device=device, dtype=dtype))

    def forward(
            self,
            x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(
            (
                einops.reduce(x ** 2, "... d_model -> ...", "sum") /
                self.d_model
            ) + self.eps
        )
        result = (x * self.weight) / einops.rearrange(rms, '... -> ... 1')
        return result.to(in_dtype)


class SwigluFFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        d_ff = d_ff or math.floor(8 * d_model / 3)
        if d_ff % 64 != 0:
            raise ValueError("Internal dimension of the FFN should be a multiple of 64")
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(
            self,
            x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        w1_x = self.w1(x)
        w3_x = self.w3(x)
        return self.w2((w1_x * torch.sigmoid(w1_x)) * w3_x)


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        sequence_positions = torch.arange(max_seq_len, device=device)
        emb_positions = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device) / d_k)
        )
        angles = einops.einsum(sequence_positions, emb_positions, "i, j -> i j")
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotations = einops.rearrange(
            torch.stack([cos, -sin, sin, cos], dim=-1),
            "... (r c) -> ... r c", r=2, c=2
        )
        self.register_buffer("rotations", rotations, persistent=False)
        self.max_seq_len = max_seq_len

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
            "... seq_len d two, ... seq_len d two1 two -> ... seq_len d two1")
        return einops.rearrange(rotated, "... seq_len d two -> ... seq_len (d two)")


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None
    ) -> Float[torch.Tensor, " ... seq_len d_out"]:
        wq_x = self.q_proj(x)
        wk_x = self.k_proj(x)
        wv_x = self.v_proj(x)
        wq_x = einops.rearrange(
            wq_x, "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads
        )
        wk_x = einops.rearrange(
            wk_x, "... seq_len (h d_k) -> ... h seq_len d_k",
            h=self.num_heads
        )
        wv_x = einops.rearrange(
            wv_x, "... seq_len (h d_v) -> ... h seq_len d_v",
            h=self.num_heads
        )
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        if self.rope is not None and token_positions is not None:
            wq_x = self.rope(wq_x, token_positions)
            wk_x = self.rope(wk_x, token_positions)
        mha = scaled_dot_product_attention(wq_x, wk_x, wv_x, mask=mask)
        mha = einops.rearrange(mha, "... h seq_len d_v -> ... seq_len (h d_v)")
        return self.output_proj(mha)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int,
        rope=None, device=None, dtype=None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwigluFFN(
            d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
    ) -> Float[torch.Tensor, " ... seq_len d_model"]:
        _, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device)
        y = self.attn(
            self.ln1(x), token_positions=token_positions
        ) + x
        return self.ffn(self.ln2(y)) + y


class Transformer(torch.nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int,
        d_model: int, num_heads: int, d_ff: int, rope_theta: float,
        device=None, dtype=None
    ):
        super().__init__()
        rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device)
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype)
        rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, rope,
                device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(
            d_model, vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[torch.Tensor, "batch_size sequence_length"]
    ) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        emb = self.token_embeddings(in_indices)
        for layer in self.layers:
            emb = layer(emb)
        out_emb = self.ln_final(emb)
        return self.lm_head(out_emb)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    normalized = x - torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(normalized)
    sum = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum


def scaled_dot_product_attention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Float[torch.Tensor, " ... queries keys"] | None = None
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    query_key_attention = einops.einsum(
        Q, K,
        "... queries d_k, ... keys d_k -> ... queries keys"
    ) / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype, device=Q.device))
    if mask is not None:
        query_key_attention.masked_fill_(~mask, -torch.inf)
    query_key_attention = softmax(query_key_attention, dim=-1)
    return einops.einsum(
        query_key_attention, V,
        "... queries keys, ... keys d_v -> ... queries d_v"
    )
