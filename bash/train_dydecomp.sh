#!/bin/bash
#$ -M YOUR EMAIL
#$ -m be
#$ -pe smp 4
#$ -q gpu@ta-a6k-001
#$ -l gpu=1
#$ -N train_dydecomp

module load conda
conda activate dydecomp

fsync $SGE_STDOUT_PATH &

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

MODEL_SAVE_PATH=YOUR_MODEL_SAVE_PATH

python steps/train/train_dydecomp.py steps/train/train_configs.yaml \
    --decomposer_name=llama3-70b-inst-4bit \
    --decompose_policy=binary \
    --verifier_name=llama3-8b-inst \
    --verify_policy=retrieval \
    --wandb_name=decomposer-llama3-70b-verifier-llama3-8b-inst_retrieval \
    --num_workers=0 \
    --train_steps=100 \
    --timesteps_per_batch=512 \
    --max_timesteps_per_episode=20 \
    --rollout_batch_size=32 \
    --mini_batch_size=32 \
    --scale_advantage=true \
    --filter_reward=true \
    --filterreward_threshold=-0.02 \
    --clip_critic_loss=false \
    --cliprange_value=0.03 \
    --vf_coef=0.02 \
    --ent_coef=0.005 \
    --temperature=0.2 \
    --cache_decomposition=false \
    --cache_verification=false \
    --logging_interval=1 \
    --save_per_step=10 \
    --model_save_path=$MODEL_SAVE_PATH \
    --hostname=$HOSTNAME \
    --port=8888