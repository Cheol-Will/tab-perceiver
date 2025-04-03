# Specify the model from [TabNet, FTTransformer, ResNet, MLP, TabTransformer,
# Trompt, ExcelFormer, FTTransformerBucket, XGBoost, CatBoost, LightGBM]
model_type=TabNet

# Specify the task type from [binary_classification, regression,
# multiclass_classification]
task_type=binary_classification

# Specify the dataset scale from [small, medium, large]
scale=small

# Specify the dataset idx from [0, 1, ...]
idx=0

# Specify the number of AutoML search trials
num_trials=1

# Specify the path to save the results
result_path=output/results.pt

# Run hyper-parameter tuning and training of the specified model on a specified
# dataset.
python data_frame_benchmark.py --model_type $model_type\
                               --task_type $task_type\
                               --scale $scale\
                               --idx $idx\
                               --num_trials $num_trials\
                               --result_path $result_path