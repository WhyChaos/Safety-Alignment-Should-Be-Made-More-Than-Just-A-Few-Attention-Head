# # dropout include anchor
# python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#   --dataset_name="safety_augmentation" --model_family="llama2" \
#   --learning_rate=2e-5 \
#   --per_device_train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.005" \
#   --logging_steps=1 \
#   --num_train_epochs=10 \
#   --gradient_checkpointing \
#   --report_to=none \
#   --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
#   --save_strategy='no' \
#   --sft_type="sft" \
#   --use_anchor=True \
#   --anchor_batch_size_per_device=16 \
#   --safety_augmentation=False \
#   --use_warmup=False \
#   --use_component_level_dropout=True \
#   --component_level_dropout_rate=0.005


# python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#   --dataset_name="safety_augmentation" --model_family="llama2" \
#   --learning_rate=2e-5 \
#   --per_device_train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01" \
#   --logging_steps=1 \
#   --num_train_epochs=10 \
#   --gradient_checkpointing \
#   --report_to=none \
#   --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
#   --save_strategy='no' \
#   --sft_type="sft" \
#   --use_anchor=True \
#   --anchor_batch_size_per_device=16 \
#   --safety_augmentation=False \
#   --use_warmup=False \
#   --use_component_level_dropout=True \
#   --component_level_dropout_rate=0.01


# python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
#   --dataset_name="safety_augmentation" --model_family="llama2" \
#   --learning_rate=2e-5 \
#   --per_device_train_batch_size=4 \
#   --gradient_accumulation_steps=1 \
#   --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.05" \
#   --logging_steps=1 \
#   --num_train_epochs=10 \
#   --gradient_checkpointing \
#   --report_to=none \
#   --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
#   --save_strategy='no' \
#   --sft_type="sft" \
#   --use_anchor=True \
#   --anchor_batch_size_per_device=16 \
#   --safety_augmentation=False \
#   --use_warmup=False \
#   --use_component_level_dropout=True \
#   --component_level_dropout_rate=0.05

# dropout exclude anchor
python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
  --dataset_name="safety_augmentation" --model_family="llama2" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.005_skip_anchor" \
  --logging_steps=1 \
  --num_train_epochs=10 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --use_anchor=True \
  --anchor_batch_size_per_device=16 \
  --safety_augmentation=False \
  --use_warmup=False \
  --use_component_level_dropout=True \
  --component_level_dropout_rate=0.005 \
  --use_skip_anchor_dropout=True


python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
  --dataset_name="safety_augmentation" --model_family="llama2" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.01_skip_anchor" \
  --logging_steps=1 \
  --num_train_epochs=10 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --use_anchor=True \
  --anchor_batch_size_per_device=16 \
  --safety_augmentation=False \
  --use_warmup=False \
  --use_component_level_dropout=True \
  --component_level_dropout_rate=0.01 \
  --use_skip_anchor_dropout=True


python finetune.py --model_name_or_path="/root/autodl-tmp/Llama-2-7b-chat-hf" \
  --dataset_name="safety_augmentation" --model_family="llama2" \
  --learning_rate=2e-5 \
  --per_device_train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --output_dir="/root/autodl-tmp/Llama-2-7b-chat-dropout0.05_skip_anchor" \
  --logging_steps=1 \
  --num_train_epochs=10 \
  --gradient_checkpointing \
  --report_to=none \
  --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True \
  --save_strategy='no' \
  --sft_type="sft" \
  --use_anchor=True \
  --anchor_batch_size_per_device=16 \
  --safety_augmentation=False \
  --use_warmup=False \
  --use_component_level_dropout=True \
  --component_level_dropout_rate=0.05 \
  --use_skip_anchor_dropout=True



shutdown -h now