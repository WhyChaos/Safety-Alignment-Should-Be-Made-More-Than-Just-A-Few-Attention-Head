
import torch
import contextlib
import functools

from typing import List, Tuple, Callable, Optional
from jaxtyping import Float
from torch import Tensor

from utils import component_utils


class SkipHookState:
    def __init__(self):
        self.is_skip: bool = False

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




def get_attn_o_proj_input_hook(head_idx: int, num_heads: int, skip_hook_state: Optional[SkipHookState] = None):
    def hook_fn(module, input):
        nonlocal head_idx
        nonlocal num_heads
        nonlocal skip_hook_state

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
        if skip_hook_state is None or skip_hook_state.is_skip is False:
            mask[:, :, start_idx:end_idx] = 0
        
        activation = activation * mask
        
        # Prevent gradient propagation through the mask part
        if skip_hook_state is not None:
            activation.register_hook(lambda grad: grad * mask)
        
        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_mlp_down_proj_input_hook(head_idx: int, num_heads: int, skip_hook_state: Optional[SkipHookState] = None):
    def hook_fn(module, input):
        nonlocal head_idx
        nonlocal num_heads
        nonlocal skip_hook_state

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
        
        if skip_hook_state is None or skip_hook_state.is_skip is False:
            mask[:, :, start_idx:end_idx] = 0
        
        activation = activation * mask
        
        # Prevent gradient propagation through the mask part
        if skip_hook_state is not None:
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
    skip_hook_state: Optional[SkipHookState] = None
):
    fwd_pre_hooks = [(model_base.layers[layer_idx].self_attn.o_proj, get_attn_o_proj_input_hook(head_idx=head_idx, num_heads=num_heads, skip_hook_state=skip_hook_state))]
    fwd_hooks = []

    return fwd_pre_hooks, fwd_hooks

def get_mlp_down_proj_hooks(
    model_base,
    layer_idx: int,
    head_idx: int,
    num_heads: int,
    skip_hook_state: Optional[SkipHookState] = None
):
    fwd_pre_hooks = [(model_base.layers[layer_idx].mlp.down_proj, get_mlp_down_proj_input_hook(head_idx=head_idx, num_heads=num_heads, skip_hook_state=skip_hook_state))]
    fwd_hooks = []

    return fwd_pre_hooks, fwd_hooks


def get_hooks(
    model,
    component_dropout_idx_list: List[int],
):
    layer_num = len(model.model.layers)
    head_num = model.model.layers[0].self_attn.num_heads
    fwd_pre_hooks, fwd_hooks = [], []
    for component_idx in component_dropout_idx_list:
        component_type, layer_idx, head_idx = component_utils.disassemble_component_idx(component_idx, layer_num, head_num)
        # attn component
        if component_type == 'attn':
            hook_pair = get_attn_o_proj_hooks(model.model, layer_idx=layer_idx, head_idx=head_idx, num_heads=head_num)
            fwd_pre_hooks += hook_pair[0]
            fwd_hooks += hook_pair[1]
        # mlp component
        elif component_type == 'mlp':
            hook_pair = get_mlp_down_proj_hooks(model.model, layer_idx=layer_idx, head_idx=head_idx, num_heads=head_num)
            fwd_pre_hooks += hook_pair[0]
            fwd_hooks += hook_pair[1]
        else:
            raise(ValueError(f"Unknown component type: {component_type}"))
    return fwd_pre_hooks, fwd_hooks