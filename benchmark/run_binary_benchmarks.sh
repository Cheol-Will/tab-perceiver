#!/bin/bash

model_type="FTPerceiver"
task_type="binary_classification"
num_trials=5

scale_types=("small" "medium" "large")

for scale_type in "${scale_types[@]}"; do
    if [ "$scale_type" == "small" ]; then
        scale_range=$(seq 0 13)
    elif [ "$scale_type" == "medium" ]; then
        scale_range=$(seq 0 8)
    elif [ "$scale_type" == "large" ]; then
        scale_range=0
    fi


    for idx in $scale_range; do
        result_path=output1/$task_type/$scale_type/$idx/$model_type.pt
        python data_frame_benchmark.py \
            --model_type "$model_type" \
            --task_type "$task_type" \
            --scale "$scale_type" \
            --idx "$idx" \
            --num_trials "$num_trials" \
            --result_path "$result_path" \
            --epoch 10
    done
done
