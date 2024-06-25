export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
MODEL_SIZE=7B
NUM_GPUS=2
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=2
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

CUDA_VISIBLE_DEVICES=3,4 accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --main_process_port 25001\
    --use_deepspeed \
    --deepspeed_config_file config/stage2_no_offloading_accelerate.conf \
    prompt_tuning.py \
    --model_name_or_path /data1/LLM_Model/Llama-2-7b-hf/ \
    --use_flash_attn \
    --tokenizer_name /data1/LLM_Model/Llama-2-7b-hf/ \
    --use_slow_tokenizer \
    --train_file data/merged_train_50k.jsonl \
    --val_file data/val_asqa.jsonl data/val_nq.jsonl data/val_hotpotqa.jsonl \
    --eval_steps 2000 \
    --max_seq_length 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 3 \
    --output_dir output/RPG_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_special_tokens \
