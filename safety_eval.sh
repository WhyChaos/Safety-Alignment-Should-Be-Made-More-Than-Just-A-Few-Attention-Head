# # baseline safety evaluation for Llama-2-7b-chat-hf
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-hf.json' \
#       --eval_template='plain'

# safety evaluation for Llama-2-7b-chat-augmented
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-augmented" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-augmented.json' \
#       --eval_template='plain'

# # safety evaluation for Llama-2-7b-chat-dropout0.05_skip_anchor
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.05_skip_anchor" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.05_skip_anchor.json' \
#       --eval_template='plain'

# # safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor.json' \
#       --eval_template='plain'

# # safety evaluation for Llama-2-7b-chat-hf-top-1
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-hf-top-1.json' \
#       --eval_template='plain' \
#       --use_component_level_dropout=True \
#       --component_dropout_idx_list 1967

# # safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor-top-1
# python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
#       --torch_dtype=bfloat16 \
#       --safety_bench='hex-phi' \
#       --model_family='llama2_base' \
#       --prompt_style='llama2_base' \
#       --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor-top-1.json' \
#       --eval_template='plain' \
#       --use_component_level_dropout=True \
#       --component_dropout_idx_list 1967


# safety evaluation for Llama-2-7b-chat-hf-top-20
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-hf-top-20.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_idx_list \
      1967 1893 1950 2009 2044 \
      1852 2030 1920 1986 1976 \
      1936 1988 1958 2022 2040 \
      1926 2042 1980 1966 1912

# safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor-top-20
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor-top-20.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_idx_list \
      1967 2030 2022 1976 1910 \
      2042 1960 1966 1958 1988 \
      1640 1920 1890 2044 1926 \
      2040 1936 1822 1754 1986