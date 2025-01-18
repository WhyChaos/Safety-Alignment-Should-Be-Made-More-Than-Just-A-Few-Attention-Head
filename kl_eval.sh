python eval_kl.py \
    --torch_dtype=bfloat16 \
    --dataset=hex-phi \
    --model_name_or_path=/root/autodl-tmp/Llama-2-7b-chat-hf \
    --save_path=results/kl/hex-phi/Llama-2-7b-chat-hf.json

python eval_kl.py \
    --torch_dtype=bfloat16 \
    --dataset=sampled_330_alpaca_anchor \
    --model_name_or_path=/root/autodl-tmp/Llama-2-7b-chat-hf \
    --save_path=results/kl/sampled_330_alpaca_anchor/Llama-2-7b-chat-hf.json

shutdown -h now