python eval_safety.py --model_name_or_path="/root/workspace/Llama-2-7b-chat-attn-head-dropout-0.5-GPU1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='new_results/dropout5gpu1/llama2_chat_no_prefilling.json' \
      --eval_template='null' 

python eval_safety.py --model_name_or_path="/root/workspace/Llama-2-7b-chat-attn-head-dropout-0.5-GPU1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='new_results/dropout5gpu1/llama2_chat_prefilled_5_harmful_tokens.json' \
      --eval_template='null' \
      --num_perfix_tokens=5

python eval_safety.py --model_name_or_path="/root/workspace/Llama-2-7b-chat-attn-head-dropout-0.5-GPU1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='new_results/dropout5gpu1/llama2_chat_prefilled_10_harmful_tokens.json' \
      --eval_template='null' \
      --num_perfix_tokens=10

python eval_safety.py --model_name_or_path="/root/workspace/Llama-2-7b-chat-attn-head-dropout-0.5-GPU1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='new_results/dropout5gpu1/llama2_chat_prefilled_20_harmful_tokens.json' \
      --eval_template='null' \
      --num_perfix_tokens=20

python eval_safety.py --model_name_or_path="/root/workspace/Llama-2-7b-chat-attn-head-dropout-0.5-GPU1" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='new_results/dropout5gpu1/llama2_chat_prefilled_40_harmful_tokens.json' \
      --eval_template='null' \
      --num_perfix_tokens=40