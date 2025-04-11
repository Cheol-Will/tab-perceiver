import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_result():
    # task_types = ["binary_classification", "regression", "multiclass_classification"]
    task_type = "binary_classification"
    scale_types = ["small", "medium", "large"]
    model_types = ["LightGBM", "FTPerceiver", "FTTransformer", "TabNet", "ResNet"]

    result_list = []

    for model_type in model_types:
        for scale_type in scale_types:
            if scale_type == "small":
                index_list = range(0, 14)
            elif scale_type == "medium":
                index_list = range(0, 9)
            else:  # large
                index_list = range(1)

            for idx in index_list:
                path = f"output/{task_type}/{scale_type}/{idx}/{model_type}.pt"

                if os.path.exists(path):
                    result = torch.load(path, weights_only=False)
                    result_list.append({
                        "dataset_index": f"{scale_type}-{idx}",
                        "scale_type": scale_type,
                        "idx": idx,
                        "model_type": model_type,
                        "best_test_metric": result["best_test_metric"]
                    })

    return result_list, task_type

def plot_result(result_list, task_type):
    result_df = pd.DataFrame(result_list, columns=["dataset_index", "model_type", "best_test_metric"])

    # columns = LightGBM, FTPerceiver, FTTransformer, so on.
    pivot_df = result_df.pivot(index="dataset_index", columns="model_type", values="best_test_metric") 
    model_cols = list(pivot_df.columns)

    if "LightGBM" not in model_cols:
        raise ValueError("LightGBM is not included.")

    model_types = [model_type for model_type in model_cols if model_type != "LightGBM"]

    plt.figure(figsize=(8, 8))
    x = np.linspace(0.5, 1.0, 500)
    y = x
    plt.fill_between(x, y, 1.0, color='blue', alpha=0.1)  # LightGBM better
    plt.fill_between(x, 0.5, y, color='red', alpha=0.1) # Deep model better
    plt.plot([0.5, 1.0], [0.5, 1.0], 'r--')

    for model_type in model_types:
        plt.scatter(
            pivot_df[model_type],
            pivot_df["LightGBM"],
            label=model_type,
            s=60,
            alpha=0.8
        )

    plt.text(0.52, 0.97, "LightGBM Better", fontsize=24, fontweight='bold', color='blue')
    plt.text(0.72, 0.53, "Deep model Better", fontsize=24, fontweight='bold', color='red')

    plt.xlabel("ROC-AUC for deep tabular models", fontsize=24)
    plt.ylabel("ROC-AUC for LightGBM", fontsize=24)
    plt.title(f"Deep Models vs LightGBM across {task_type} Task", fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = f"output/plots/deep_vs_lightgbm_{task_type}.png"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300)
    plt.show()