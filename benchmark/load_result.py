import os
import torch


model_type = "FTPerceiver"
task_type = "binary_classification"
scale_types = ["small" "medium" "large"]


model_types = ["FTPerceiver", "FTTransformer"]
task_types = ["binary_classification", "regression", "multiclass_classification"]


for model_type in model_types:
    for task_type in task_types:
        path = f"output/{task_type}-0/{model_type}.pt"
        if os.path.exists(path):
            result = torch.load(path, weights_only=False)
            print(result)

        else: 
            continue


path = "output/FTPerceiver.pt"
result = torch.load(path, weights_only=False)
print(result)