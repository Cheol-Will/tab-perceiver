import os
import torch



def load_result():
    model_type = "FTPerceiver"
    task_type = "binary_classification"
    scale_types = ["small", "medium", "large"]

    # model_types = ["FTPerceiver", "FTTransformer"]
    # task_types = ["binary_classification", "regression", "multiclass_classification"]
    result_dict = {"small": [], "medium": [], "large": []}
    for scale_type in scale_types:
        if scale_type == "small":
            index_list = list(range(0, 14))
        elif scale_type == "medium":
            index_list = list(range(0, 9))
        else: 
            index_list = list(range(1))

        for idx in index_list:
            if os.path.exists(path):
                path=f"output/{task_type}/{scale}/{idx}/{model_type}.pt"
                result = torch.load(path, weights_only=False)  
                result_dict[scale_type].append(result)
            else:
                continue
    return result_dict