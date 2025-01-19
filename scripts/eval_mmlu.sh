llamafactory-cli eval \
    --model_name_or_path /root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor \
    --task mmlu_test \
    --lang en \
    --template llama2  \
    --n_shot 5 \
    --trust_remote_code \
    --save_dir eval/mmlu/Llama-2-7b-chat-dropout0.01_skip_anchor