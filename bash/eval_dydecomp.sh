#!/bin/bash
#$ -M YOUR EMAIL
#$ -m be
#$ -pe smp 1
#$ -q gpu@ta-a6k-001
#$ -l gpu=1
#$ -N eval_dydecomp

module load conda
conda activate dydecomp

fsync $SGE_STDOUT_PATH &

MODEL_SAVE_PATH=YOUR_MODEL_SAVE_PATH

for dataset in ChatGPT PerplexityAI
do
    for atomicity in 4 3 2 1 0
    do
        python steps/eval/eval_ppo.py steps/eval/eval_configs.yaml \
            --saved_model_path=$MODEL_SAVE_PATH/best_model \
            --decomposer_name=llama3-70b-inst-4bit \
            --verifier_name=llama3-8b-inst \
            --verify_policy=retrieval \
            --test_data_path=data/FactScore/$dataset/atomicity_$atomicity \
            --result_save_dir=results/FactScore/$dataset \
            --tag=main \
            --atomicity=$atomicity \
            --hostname=$HOSTNAME \
            --port=8888
    done
done