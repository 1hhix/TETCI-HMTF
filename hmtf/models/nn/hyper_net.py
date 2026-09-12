from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.models.nn.attention import MultiHeadAttention
from rl4co.models.nn.mlp import MLP
from rl4co.models.nn.moe import MoE
from rl4co.utils.pylogger import get_pylogger
from torch import Tensor

log = get_pylogger(__name__)


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
                "rms": RMSNorm,
            }.get(normalization, None)
            self.normalizer = (
                normalizer_class(embed_dim, affine=True)
                if normalizer_class is not None
                else None
            )
        else:
            self.normalizer = "layer"
        if self.normalizer is None:
            log.error(
                "Normalization type {} not found. Skipping normalization.".format(
                    normalization
                )
            )

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        elif isinstance(self.normalizer, RMSNorm):
            return self.normalizer(x)
        else:
            assert self.normalizer is None, "Unknown normalizer type {}".format(
                self.normalizer
            )
            return x


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate_back(x, shift, scale):
    return (x-shift.unsqueeze(1)) / (1 + scale.unsqueeze(1))


import torch.nn.functional as F
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x


class TaskEncoder(nn.Module):
    def __init__(self, embed_dim, num_outputs):
        super().__init__()
        self.global_projector =  nn.Sequential(
            nn.Linear(4*embed_dim, 2 * embed_dim, bias=True),
            nn.SiLU(),
        )
        self.output = nn.ModuleList([nn.Linear(2 * embed_dim, embed_dim) for _ in range(num_outputs)])
    def forward(self, c):
        global_task_feature = self.global_projector(c)
        return [decoder(global_task_feature) for decoder in self.output]

class MultiHeadModulation(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    def forward(self, x, shift,scale):
        batch_size, seq_len, _ = x.size()
        shift = shift.view(batch_size, 1, self.num_heads, -1)
        scale = scale.view(batch_size, 1, self.num_heads, -1)

        # Reshape input tensor x to [batch_size, seq_len, num_heads, embed_dim // num_heads]
        x = x.view(batch_size, seq_len, self.num_heads, -1)

        # Apply modulation: modulated = x * (1 + scale) + shift
        modulated = x * (1 + scale) + shift

        return modulated.view(batch_size, seq_len, self.embed_dim)


def multi_head_attention(query, key, value, num_heads):
    """
    Compute Multi-Head Attention (MHA).

    Args:
        query: Tensor of shape (b, seq_len, embedding)
        key: Tensor of shape (b, 1, embedding)
        value: Tensor of shape (b, 1, embedding)
        num_heads: Number of attention heads.

    Returns:
        Tensor of shape (b, seq_len, embedding), the MHA output.
    """
    b, seq_len, embedding_dim = query.shape
    assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."

    head_dim = embedding_dim // num_heads  # Dimension per head

    # Linear transformations for query, key, and value
    query = query.view(b, seq_len, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, seq_len, head_dim)
    key = key.view(b, 1, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, 1, head_dim)
    value = value.view(b, 1, num_heads, head_dim).transpose(1, 2)  # (b, num_heads, 1, head_dim)

    # Scaled dot-product attention
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)  # (b, num_heads, seq_len, 1)
    attention_weights = F.softmax(scores, dim=-2)  # (b, num_heads, seq_len, 1)

    attention_output = torch.matmul(attention_weights, value)  # (b, num_heads, seq_len, head_dim)

    # Combine attention output from all heads
    attention_output = attention_output.transpose(1, 2).contiguous()  # (b, seq_len, num_heads, head_dim)
    attention_output = attention_output.view(b, seq_len, embedding_dim)  # (b, seq_len, embedding_dim)

    return attention_output
class Hyper_Block_T(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        feedforward_hidden: Optional[int] = None,  # if None, use 4 * embed_dim
        normalization: Optional[str] = "instance",
        use_prenorm: bool = False,
        bias: bool = True,
        sdpa_fn: Optional[Callable] = None,
        moe_kwargs: Optional[dict] = None,
        parallel_gated_kwargs: Optional[dict] = None,
    ):
        super(Hyper_Block_T, self).__init__()
        feedforward_hidden = (
            4 * embed_dim if feedforward_hidden is None else feedforward_hidden
        )
        num_neurons = [feedforward_hidden] if feedforward_hidden > 0 else []
        if moe_kwargs is not None:
            ffn = MoE(embed_dim, embed_dim, num_neurons=num_neurons, **moe_kwargs)
        elif parallel_gated_kwargs is not None:
            ffn = ParallelGatedMLP(embed_dim, **parallel_gated_kwargs)
        else:
            ffn = MLP(
                input_dim=embed_dim,
                output_dim=embed_dim,
                num_neurons=num_neurons,
                hidden_act="GELU",# ReLU
            )
        self.norm_attn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, bias=bias, sdpa_fn=sdpa_fn
        )
        self.norm_ffn = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.norm = (
            Normalization(embed_dim, normalization)
            if normalization is not None
            else lambda x: x
        )
        self.ffn = ffn
        self.use_prenorm = use_prenorm
        self.hyper_param=TaskEncoder(embed_dim, 6)
        self.adaLN_modulation = TaskEncoder(embed_dim, 6)
        self.MHM=MultiHeadModulation(embed_dim, 8)


    def forward(self, h: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if self.use_prenorm:
            x,c=h
            shift_msa, scale_msa, shift_mlp, scale_mlp,gate_msa,gate_mlp = self.hyper_param(c)
            x = x +self.attention(self.norm_attn(modulate(x, shift_msa, scale_msa)))
            x = x + self.ffn(self.norm_ffn(modulate(x, shift_mlp, scale_mlp)))
            x = x + self.norm(modulate(x, gate_msa, gate_mlp))
            return (x,c)
        else:
            x,c=h
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.hyper_param(c).chunk(6, dim=1)
            x = x + self.norm_attn(self.attention(modulate(x, shift_msa, scale_msa)) * gate_msa.unsqueeze(1))
            x = x + self.norm_ffn(self.ffn(modulate(x, shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1))
            return (x,c)
