import collections
import logging
from typing import Optional, Tuple, Type, Union

log = logging.getLogger(__name__)

try:
    import fbgemm_gpu.experimental.gen_ai

    log.info("Using efficient FP8 or INT4 operators in FBGEMM.")
except ImportError:
    log.error("No efficient FP8 or INT4 operators. Please install FBGEMM.")
    raise

import torch
from torch import nn, Tensor


class Fp8ScaledWeights:
    @property
    def __class__(self) -> Type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None

class Fp8RowwiseWeights(
    Fp8ScaledWeights,
    collections.namedtuple(
        "Fp8RowwiseWeights",
        ["weight", "scale", "shape", "activation_scale_ub"],
    ),
):
    pass


class Int4ScaledWeights:
    @property
    def __class__(self) -> Type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None

class Int4Weights(
    Int4ScaledWeights,
    collections.namedtuple(
        "Int4Weights",
        ["weight", "scale", "shape"],
    ),
):
    pass


def int4_row_quantize(
    x: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
   
    n_bit = 4 
    to_quant = x.reshape(-1, group_size).to(torch.float)

    max_val = torch.abs(to_quant).amax(dim=1, keepdim=True)
    max_int = 2 ** (n_bit - 1)
    min_int = -(2 ** (n_bit - 1))
    scales = max_val.clamp(min=1e-6) / max_int

    out = to_quant.div(scales).round().clamp_(min_int, max_int - 1)

    out = out.to(dtype=torch.int8).reshape(x.shape)
    scales = scales.view(x.shape[0], -1).t().contiguous()

    return out, scales


def pack_int4(x: torch.Tensor) -> torch.Tensor:
    low_x = x[:, ::2]
    high_x = x[:, 1::2]
    high_x = torch.bitwise_left_shift(high_x, 4)
    low_x = torch.bitwise_and(low_x, 0xF)

    return torch.bitwise_or(low_x, high_x).contiguous()

def bmm_nt(
    x: Tensor,
    w: Union[Fp8RowwiseWeights, Int4Weights],
    num_tokens: Optional[Tensor] = None,
) -> Tensor:
    if isinstance(w, Fp8ScaledWeights):
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, w.activation_scale_ub)
        return torch.ops.fbgemm.f8f8bf16_rowwise_batched(xq, w.weight, x_scale, w.scale)
    elif isinstance(w, Int4ScaledWeights):
        return torch.ops.fbgemm.bf16i4bf16_rowwise_batched(x, w.weight, w.scale, torch.zeros_like(w.scale))
    else:
        raise ValueError("Unsupported quantization type")


def ffn_swiglu(
    x: Tensor,
    w1: Union[Fp8RowwiseWeights, Int4Weights],
    w3: Union[Fp8RowwiseWeights, Int4Weights],
    w2: Union[Fp8RowwiseWeights, Int4Weights],
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    if isinstance(w1, Fp8ScaledWeights) and isinstance(w3, Fp8ScaledWeights) and isinstance(w2, Fp8ScaledWeights):
        return ffn_swiglu_dynamic(x, w1, w3, w2, w1.activation_scale_ub, num_tokens, is_memory_bounded)
    elif isinstance(w1, Int4ScaledWeights) and isinstance(w3, Int4ScaledWeights) and isinstance(w2, Int4ScaledWeights):
        return ffn_swiglu_dynamic(x, w1, w3, w2, None, num_tokens, is_memory_bounded)

    (B, T, D) = x.shape  # noqa: N806
    (HD_L, D_) = w1.shape  # noqa: N806
    assert D_ == D

    assert isinstance(w1, Tensor)
    assert isinstance(w3, Tensor)
    x1 = x.view(B * T, D) @ w1.T
    x2 = x.view(B * T, D) @ w3.T
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2
    assert isinstance(w2, Tensor)
    return (z @ w2.T).view(B, T, D)


@torch.inference_mode()
def quantize_fp8(
    w: Tensor,
    fp8_activation_scale_ub: float,
    output_device: Optional[torch.device] = None,
) -> Fp8RowwiseWeights:
    
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
    del w
    return Fp8RowwiseWeights(
        weight=wq,
        scale=w_scale,
        shape=wq.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def quantize_int4(
    w: Tensor,
    output_device: Optional[torch.device] = None,
) -> Int4Weights:
    """Quantize [n, k/2] weight tensor.

    Args:
        w (Tensor): [n, k/2] input high precision tensor to quantize.
    """
    if w.ndim >= 3:
        wq, scale = zip(*[int4_row_quantize(i) for i in w])
        wq = torch.stack([pack_int4(i) for i in wq], dim=0)
        scale = torch.stack(scale, dim=0)
    else:
        wq, scale = int4_row_quantize(w)
        wq = pack_int4(wq)
    del w
    return Int4Weights(
        weight=wq.to(output_device),
        scale=scale.to(output_device),
        shape=wq.shape,
    )


@torch.inference_mode()
def load_fp8(
    w: Tensor,
    w_scale: Tensor,
    fp8_activation_scale_ub: float,
    output_device: Optional[torch.device] = None,
) -> Fp8RowwiseWeights:
  
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device=output_device,
    )
    return Fp8RowwiseWeights(
        weight=w.to(torch.float8_e4m3fn).to(device=output_device),
        scale=w_scale.to(device=output_device),
        shape=w.shape,
        activation_scale_ub=activation_scale_ub,
    )


@torch.inference_mode()
def load_int4(
    w: Tensor,
    scale: Tensor,
    output_device: Optional[torch.device] = None,
) -> Int4Weights:
  Int4Weights(
        weight=w.to(torch.int8).to(device=output_device),
        scale=scale.to(device=output_device),
        shape=w.shape,
    )


def fc_dynamic(
    x: Tensor,
    w: Union[Fp8RowwiseWeights, Int4Weights],
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
  
    if isinstance(w, Int4Weights):
        y = torch.ops.fbgemm.bf16i4bf16_rowwise(x, w.weight, w.scale, torch.zeros_like(w.scale))
    else:
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(x, num_tokens, activation_scale_ub)
        y = torch.ops.fbgemm.f8f8bf16_rowwise(xq, w.weight, x_scale, w.scale, use_fast_accum=True)
        del xq
    return y


def ffn_swiglu_dynamic(
    x: Tensor,
    w1: Union[Fp8RowwiseWeights, Int4Weights],
    w3: Union[Fp8RowwiseWeights, Int4Weights],
    w2: Union[Fp8RowwiseWeights, Int4Weights],
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    assert x.dim() == 3 or x.dim() == 2
    if x.dim() == 3:
        (B, T, D) = x.shape
    else:
        (T, D) = x.shape
        B = 1

    HD_L = w1.shape[0]
    assert HD_L == w3.shape[0]
    x1 = fc_dynamic(
        x.view(B * T, D),
        w1,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    x2 = fc_dynamic(
        x.view(B * T, D),
        w3,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2

    z_ = fc_dynamic(z, w2, activation_scale_ub, num_tokens, is_memory_bounded)

    if x.dim() == 3:
        return z_.view(B, T, D)
    else:
        return z_
