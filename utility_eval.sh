# baseline

# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-hf' \
#       --dataset='sql_create_context' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/sql_create_context_Llama-2-7b-chat-hf.json"


python eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-hf' \
      --dataset='gsm8k' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='gsm8k' \
      --save_path="logs/fine-tuning-attack/utility_eval/gsm8k_Llama-2-7b-chat-hf-test2.json"


# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-hf' \
#       --dataset='samsum' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/samsum_Llama-2-7b-chat-hf.json"

# Llama-2-7b-chat-dropout0.01_skip_anchor
# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor' \
#       --dataset='sql_create_context' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/sql_create_context_Llama-2-7b-chat-dropout0.01_skip_anchor.json"


python eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor' \
      --dataset='gsm8k' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='gsm8k' \
      --save_path="logs/fine-tuning-attack/utility_eval/gsm8k_Llama-2-7b-chat-dropout0.01_skip_anchor-test2.json"


# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor' \
#       --dataset='samsum' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/samsum_Llama-2-7b-chat-dropout0.01_skip_anchor.json"


# Llama-2-7b-chat-dropout0.05_skip_anchor
# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.05_skip_anchor' \
#       --dataset='sql_create_context' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/sql_create_context_Llama-2-7b-chat-dropout0.05_skip_anchor.json"


python eval_utility.py \
      --torch_dtype=bfloat16 \
      --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.05_skip_anchor' \
      --dataset='gsm8k' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='gsm8k' \
      --save_path="logs/fine-tuning-attack/utility_eval/gsm8k_Llama-2-7b-chat-dropout0.05_skip_anchor-test2.json"


# python eval_utility.py \
#       --torch_dtype=bfloat16 \
#       --model_name_or_path='/root/autodl-tmp/Llama-2-7b-chat-dropout0.05_skip_anchor' \
#       --dataset='samsum' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='rouge_1' \
#       --save_path="logs/fine-tuning-attack/utility_eval/samsum_Llama-2-7b-chat-dropout0.05_skip_anchor.json"

shutdown -h now