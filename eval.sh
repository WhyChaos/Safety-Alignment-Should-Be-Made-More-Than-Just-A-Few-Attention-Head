# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi_with_harmful_prefix' \
#       --model_family='llama2' \
#       --prompt_style='llama2' \
#       --evaluator='none' \
#       --save_path='logs/data_augmentation/llama2_chat_prefilled_10_harmful_tokens.json' \
#       --eval_template='null' \
#       --num_perfix_tokens=10


python eval_safety.py --model_name_or_path="logs/data_augmentation/Llama-2-7b-chat-augmented" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi_with_harmful_prefix' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='none' \
      --save_path='logs/data_augmentation/llama2_chat_augmented_prefilled_10_harmful_tokens.json' \
      --eval_template='null' \
      --num_perfix_tokens=10 