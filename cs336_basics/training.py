import math
from typing import Callable, Iterable, Optional
import torch
from torch import Tensor
from jaxtyping import Float, Int


def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    normalized = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    exp = torch.exp(normalized)
    sum = torch.sum(exp, dim=-1)
    losses = torch.log(sum) - normalized[torch.arange(len(targets)), targets]
    return torch.mean(losses)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        for beta in betas:
            if not 0 <= beta <= 1.0:
                raise ValueError(f"beta should be in range <0,1>")
        for param_name, param in [("lr", lr), ("eps", eps), ("weight_decay", weight_decay)]:
            if param < 0:
                raise ValueError(f"{param_name} should be non-negative")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                if "first_moment" not in state:
                    state["first_moment"] = torch.zeros_like(p.data)
                if "second_moment" not in state:
                    state["second_moment"] = torch.zeros_like(p.data)

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                first_moment.mul_(beta_1).add_((1 - beta_1) * p.grad)
                second_moment.mul_(beta_2).add_((1 - beta_2) * (p.grad ** 2))
                lr_t = lr * (math.sqrt(1 - beta_2 ** t) / (1 - beta_1 ** t))
                p.data.sub_(lr_t * (first_moment / (torch.sqrt(second_moment) + eps)))
                p.data.sub_(lr * weight_decay * p.data)
                state["t"] = t + 1
        return loss


def cosine_lr_schedule(t: int, lr_min: float, lr_max: float, warm_up_steps: int, cosine_steps: int) -> float:
    if t < warm_up_steps:
        return (t * lr_max) / warm_up_steps
    elif t <= cosine_steps:
        angle = ((t - warm_up_steps) / (cosine_steps - warm_up_steps)) * math.pi
        return lr_min + ((1 + math.cos(angle)) / 2) * (lr_max - lr_min)
    else:
        return lr_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = (p.grad for p in parameters if p.grad is not None)
    grads_stacked = torch.cat([grad.view(-1) for grad in grads])
    grad_l2 = torch.linalg.vector_norm(grads_stacked)
    if grad_l2 >= max_l2_norm:
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(max_l2_norm).div_(grad_l2 + eps)
