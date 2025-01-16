
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()




def get_attn_o_proj_input_hook(head_idx: int, num_heads: int):
    def hook_fn(module, input):
        nonlocal head_idx
        nonlocal num_heads

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # Calculate the dimension of each head
        head_dim = activation.size(-1) // num_heads
        # Mask the output of the specified head
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        
        mask = torch.ones_like(activation)
        mask.requires_grad_(False)
        mask[:, :, start_idx:end_idx] = 0
        
        activation = activation * mask
        
        # Prevent gradient propagation through the mask part
        activation.register_hook(lambda grad: grad * mask)
        
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_mlp_down_proj_input_hook(head_idx: int, num_heads: int):
    def hook_fn(module, input):
        nonlocal head_idx
        nonlocal num_heads

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        # Calculate the dimension of each head
        head_dim = activation.size(-1) // num_heads
        # Mask the output of the specified head
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim
        
        mask = torch.ones_like(activation)
        mask.requires_grad_(False)
        mask[:, :, start_idx:end_idx] = 0
        
        activation = activation * mask
        
        # Prevent gradient propagation through the mask part
        activation.register_hook(lambda grad: grad * mask)
        
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn



def get_attn_o_proj_hooks(
    model_base,
    layer_idx: int,
    head_idx: int,
    num_heads: int,
):
    fwd_pre_hooks = [(model_base.layers[layer_idx].self_attn.o_proj, get_attn_o_proj_input_hook(head_idx=head_idx, num_heads=num_heads))]
    fwd_hooks = []

    return fwd_pre_hooks, fwd_hooks

def get_mlp_down_proj_hooks(
    model_base,
    layer_idx: int,
    head_idx: int,
    num_heads: int,
):
    fwd_pre_hooks = [(model_base.layers[layer_idx].mlp.down_proj, get_mlp_down_proj_input_hook(head_idx=head_idx, num_heads=num_heads))]
    fwd_hooks = []

    return fwd_pre_hooks, fwd_hooks
