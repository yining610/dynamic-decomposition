for dataset in ChatGPT PerplexityAI
do
    for atomicity in -1 0 1 2 3 4
    do
        echo "Preparing data at atomicity: $atomicity"
        python steps/prep/factscore_data_prep.py \
            --data-save-dir data/FactScore/$dataset \
            --custom-split-path data/FactScore/$dataset/custom_split \
            --atomicity $atomicity
    done
done