#!/bin/bash
#$ -M YOUR EMAIL
#$ -m be
#$ -pe smp 1
#$ -q gpu@ta-a6k-001
#$ -l gpu=1
#$ -N pilot_experiments

conda activate dydecomp

fsync $SGE_STDOUT_PATH &

for dataset in ChatGPT PerplexityAI
do
    for atomicity in -1 0 1 2 3 4
    do
        for policy_name in retrieval demonstration non-context
        do
            for model_name in ft-t5-3B inst-llama-7B llama3-8b-inst
            do
                echo "Running dataset: $dataset, atomicity: $atomicity, policy: $policy_name, model: $model_name"
                python steps/pilot_experiments/atomicity_test.py \
                    --atomicity $atomicity \
                    --policy-name $policy_name \
                    --model-name $model_name \
                    --write-to results/pilot_experiments/$dataset \
                    --data-dir data/FactScore/$dataset
            done
        done
    done
done