model_types=("FTPerceiver" "FTTransformer")
task_types=("multiclass_classification" "binary_classification" "regression") 

scale=medium
idx=0
num_trials=1

for model_type in "${model_types[@]}"; do
    for task_type in "${task_types[@]}"; do
        result_path=output/$task_type/$scale/$idx/$model_type.pt
        python data_frame_benchmark.py --model_type $model_type\
                                       --task_type $task_type\
                                       --scale $scale\
                                       --idx $idx\
                                       --num_trials $num_trials\
                                       --result_path $result_path\
                                       --epoch 10
    done
done