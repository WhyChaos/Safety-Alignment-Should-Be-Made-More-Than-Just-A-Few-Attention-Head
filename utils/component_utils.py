from typing import Tuple

def disassemble_component_idx(component_idx: int, layer_num, 
                              head_num) -> Tuple[str, int, int]:
    assert component_idx < layer_num * head_num * 2
    if component_idx % 2 == 0:
        layer_idx = component_idx // (head_num * 2)
        head_idx = (component_idx // 2) % head_num
        return 'attn', layer_idx, head_idx
    else:
        layer_idx = component_idx // (head_num * 2)
        head_idx = (component_idx // 2) % head_num
        return 'mlp', layer_idx, head_idx
    