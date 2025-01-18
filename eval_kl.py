from transformers import BitsAndBytesConfig
from tqdm import tqdm
from transformers import HfArgumentParser
from trl import ModelConfig, get_kbit_device_map, get_peft_config, get_quantization_config
from dataclasses import dataclass, field
import torch
import time
import os
from finetuning_buckets.models import get_model
from finetuning_buckets.inference.kl_eval import evaluator
from datasets import set_caching_enabled
set_caching_enabled(False)


@dataclass
class ScriptArguments:

    dataset: str = field(default="hex-phi", metadata={"help": "the dataset to evaluate"})
    model_family: str = field(default="llama2", metadata={"help": "the model family"})
    prompt_style: str = field(default="llama2", metadata={"help": "the string prompt style"})
    save_path: str = field(default=None, metadata={"help": "the save path"})
    batch_size_per_device: int = field(default=32, metadata={"help": "the batch size"})
    

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments, ModelConfig))
    args, model_config = parser.parse_args_into_dataclasses()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    
    print(f"torch_dtype: {torch_dtype}")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )


    ################
    # Model & Tokenizer
    ################

    model, tokenizer = get_model.get_model(model_config.model_name_or_path, model_kwargs, model_family=args.model_family, padding_side="left")
    model.eval()

    start_time = time.time()
    evaluator.eval_in_batch(model, args.prompt_style, tokenizer, save_path = args.save_path, batch_size_per_device = args.batch_size_per_device,
                bench = args.dataset)
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open(f"{os.path.dirname(args.save_path)}/timing_results.txt", "w") as file:
        file.write(f"Elapsed time: {elapsed_time / 60} minutes\n")