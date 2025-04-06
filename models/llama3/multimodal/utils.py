import collections
import torch

def get_negative_inf_value(dtype):
    return torch.finfo(dtype).min

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)
