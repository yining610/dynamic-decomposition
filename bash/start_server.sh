#!/bin/bash
#$ -M YOUR EMAIL
#$ -m be
#$ -pe smp 1
#$ -q gpu@ta-a6k-001
#$ -l gpu=1
#$ -N vllm_server

module load conda
conda activate dydecomp

fsync $SGE_STDOUT_PATH &

MODEL_CACHE_DIR=YOUR_MODEL_CACHE_DIR

python -m vllm.entrypoints.openai.api_server \
    --model casperhansen/llama-3-70b-instruct-awq \
    --download_dir $MODEL_CACHE_DIR \
    --tensor_parallel_size 1 \
    --quantization awq \
    --gpu_memory_utilization 0.95 \
    --max_model_len 4096 \
    --host $HOSTNAME \
    --port 8888 \
    --disable-log-requests
