from finetuning_buckets.datasets.utils import get_eval_data
from finetuning_buckets.inference import chat
import json
from tqdm import tqdm
import numpy as np
import random
import re
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from accelerate.state import PartialState
import torch
import os
from utils import component_utils


class MyDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

from torch.utils.data._utils.collate import default_collate

def custom_collate_fn_for_labeled_data(batch):
    data_points, labels = zip(*batch)
    labels = default_collate(labels)
    return data_points, labels

def custom_collate_fn_for_unlabeled_data(batch):
    return batch



def eval_in_batch(model, prompt_style, tokenizer, save_path = None, batch_size_per_device = 16,
                bench = 'hex-phi'):
    
    accelerator = Accelerator()

    with PartialState().local_main_process_first():

        if bench == 'hex-phi':
            dataset, plain_text = get_eval_data.get_hex_phi(split='test')
        elif bench == 'sampled_330_alpaca_anchor':
            dataset, plain_text = get_eval_data.get_sampled_330_alpaca_anchor()
        else:
            raise ValueError('Benchmark {} not maintained'.format(bench))


        
        dataset = MyDataset(dataset)
    
    
    cnt = 0
    results = []

    collate_fn = custom_collate_fn_for_unlabeled_data

    dataloader_params = {
        "batch_size": batch_size_per_device,
        "collate_fn": collate_fn,
        "shuffle": False,
    }

    # prepare dataloader
    data_loader = accelerator.prepare(DataLoader(dataset, **dataloader_params))
    model = accelerator.prepare(model)
    model.eval()
    
    Generator = chat.Chat(model = model, prompt_style = prompt_style, tokenizer = tokenizer,
                         init_system_prompt = None)
    
    results = []
    for batch in tqdm(data_loader):
        with torch.inference_mode():
            batch_input_sample = batch
            

            last_kl_list = Generator.generate_kl_in_batch(inputs = batch_input_sample, accelerator = accelerator)

            accelerator.wait_for_everyone()

            results.append(last_kl_list)

    

    # gather results from all devices
    results_serialized = torch.tensor( bytearray( json.dumps(results).encode('utf-8') ), dtype=torch.uint8 ).to(accelerator.device)
    results_serialized = results_serialized.unsqueeze(0)
    results_serialized = accelerator.pad_across_processes(results_serialized, dim=1, pad_index=0)
    gathered_padded_tensors = accelerator.gather(results_serialized).cpu()


    if accelerator.is_local_main_process:

        results_all = []
        idx = 0
        for t in gathered_padded_tensors:
            idx += 1
            data = t.numpy().tobytes().rstrip(b'\x00')
            results_all += json.loads(data.decode('utf-8'))
        
        results = results_all
        results_array = np.array(results)
        combined_results = np.sum(results_array, axis=0)
        sorted_indices = np.argsort(-combined_results)
        
        layer_num = len(model.model.layers)
        head_num = model.model.layers[0].self_attn.num_heads
        results = []
        for component_idx in sorted_indices:
            component_type, layer_idx, head_idx =  component_utils.disassemble_component_idx(component_idx=component_idx, layer_num=layer_num, head_num=head_num)
            results.append({
                'component_idx': component_idx.item(),
                'component_type': component_type,
                'layer_idx': layer_idx.item(),
                'head_idx': head_idx.item(),
                'kl': combined_results[component_idx].item()
            })
        with open(save_path, 'w') as json_file:
            json.dump(results, json_file)