model_types=("FTPerceiver" "FTTransformer")
task_types=("binary_classification" "regression" "multiclass_classification")

scale=small
idx=0
num_trials=1

for model_type in "${model_types[@]}"; do
    for task_type in "${task_types[@]}"; do
        result_path=output/$task_type-$idx/$model_type.pt
        python data_frame_benchmark.py --model_type $model_type\
                                       --task_type $task_type\
                                       --scale $scale\
                                       --idx $idx\
                                       --num_trials $num_trials\
                                       --result_path $result_path