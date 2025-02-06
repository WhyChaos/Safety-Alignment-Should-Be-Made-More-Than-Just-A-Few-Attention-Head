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


# safety evaluation for Llama-2-7b-chat-hf-top-300
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-hf-top-300.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=300

# safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor-top-300
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor-top-300.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=300

# safety evaluation for Llama-2-7b-chat-hf-top-400
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-hf-top-400.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=400

# safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor-top-400
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor-top-400.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=400

# safety evaluation for Llama-2-7b-chat-hf-top-500
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-hf-top-500.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=500

# safety evaluation for Llama-2-7b-chat-dropout0.01_skip_anchor-top-500
python eval_safety.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2_base' \
      --prompt_style='llama2_base' \
      --save_path='results/safety/competition/Llama-2-7b-chat-dropout0.01_skip_anchor-top-500.json' \
      --eval_template='plain' \
      --use_component_level_dropout=True \
      --component_dropout_num=500

shutdown -h now