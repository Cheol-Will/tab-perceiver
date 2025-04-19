import os
from collections import Counter
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_result(
    dir_name : str = "output", 
    task_type : str = "binary_classification"
):
    # task_types = ["binary_classification", "regression", "multiclass_classification"]
    scale_types = ["small", "medium", "large"]
    model_types = [
        "LightGBM",
        "TabPerceiver", 
        "TabNet", 
        "FTTransformer", 
        "ResNet", 
        "TabTransformer", 
        "Trompt", 
        "ExcelFormer"
    ]

    result_dict = {model_type : [] for model_type in model_types}
    for model_type in model_types:
        for scale_type in scale_types:
            if scale_type == "small":
                index_list = range(0, 14)
            elif scale_type == "medium":
                index_list = range(0, 9)
            else:  # large
                index_list = range(1)

            for idx in index_list:
                path = os.path.join(dir_name, f"{task_type}/{scale_type}/{idx}/{model_type}.pt")

                if os.path.exists(path):
                    result = torch.load(path, weights_only=False)
                    if model_type == "LightGBM":
                        result_dict[model_type].append({
                            "dataset_index": f"{scale_type}-{idx}",
                            "scale_type": scale_type,
                            "idx": idx,
                            "model_type": model_type,
                            "best_test_metric": result["best_test_metric"],
                            "best_cfg": result["best_cfg"],
                        })
                    else:
                        result_dict[model_type].append({
                            "dataset_index": f"{scale_type}-{idx}",
                            "scale_type": scale_type,
                            "idx": idx,
                            "model_type": model_type,
                            "best_test_metric": result["best_test_metric"],
                            "best_model_cfg": result["best_model_cfg"],
                            "best_train_cfg": result["best_train_cfg"],
                        })
    return result_dict, task_type

def plot_result(
    result_dict: dict[str, list[dict]], 
    task_type: str
):
    # Flatten
    rows = []
    for model_type, entries in result_dict.items():
        for entry in entries:
            rows.append({
                "dataset_index": entry["dataset_index"],
                "model_type": model_type,
                "best_test_metric": entry["best_test_metric"]
            })
    result_df = pd.DataFrame(rows)

    # Pivot
    pivot_df = result_df.pivot(index="dataset_index", columns="model_type", values="best_test_metric")
    if "LightGBM" not in pivot_df.columns:
        raise ValueError("LightGBM is not included in result_dict.")

    # Prepare plot
    plt.figure(figsize=(9, 9))
    x = np.linspace(0.5, 1.0, 500)
    plt.fill_between(x, x, 1.0, color='blue', alpha=0.1)
    plt.fill_between(x, 0.5, x, color='red', alpha=0.1)
    plt.plot([0.5, 1.0], [0.5, 1.0], 'r--')

    for model_type in pivot_df.columns:
        if model_type == "LightGBM":
            continue
        plt.scatter(
            pivot_df[model_type],
            pivot_df["LightGBM"],
            label=model_type,
            s=240,
            alpha=0.8,
            marker="X"
        )

    plt.text(0.52, 0.97, "LightGBM Better", fontsize=24, fontweight='bold', color='blue')
    plt.text(0.72, 0.53, "Deep model Better", fontsize=24, fontweight='bold', color='red')
    plt.xlabel("ROC-AUC for deep tabular models", fontsize=32)
    plt.ylabel("ROC-AUC for LightGBM", fontsize=32)
    plt.title(f"Deep Models vs LightGBM across {task_type} Task", fontsize=20)
    plt.legend(loc='center right', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    out_path = f"output/plots/deep_vs_lightgbm_{task_type}_250418.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved plot to {out_path}")


def save_result_dataframe(
    result_dict: dict[str, list[dict]],
    task_type: str
):
    rows = []
    for model_type, entries in result_dict.items():
        for entry in entries:
            rows.append({
                "dataset_index": entry["dataset_index"],
                "model_type": model_type,
                "best_test_metric": entry["best_test_metric"]
            })
    result_df = pd.DataFrame(rows)

    # Save full results
    out_path = f"output/metric/leaderboard_{task_type}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"Saved leaderboard to {out_path}")

def plot_hyperparameter_distribution(
    result_dict: dict[str, list[dict]],
    model_type: str
):
    entries = result_dict.get(model_type, [])

    # Define search spaces
    model_search_space = {
        'num_heads': [4, 8],
        'num_layers': [4, 6, 8],
        'num_latents': [4, 8, 16, 32],
        'hidden_dim': [32, 64, 128, 256],
        'dropout_prob': [0.0, 0.2],
    }
    # model_search_space = {
    #     'num_heads': [4, 8],
    #     'num_layers': [4, 6, 8],
    #     'num_latent_array': [4, 8, 16, 32],
    #     'latent_channels': [32, 64, 128, 256],
    #     'dropout_prob': [0.0, 0.2],
    # }
    train_search_space = {
        'batch_size': [128, 256],
        'base_lr': [0.0001, 0.001],
        'gamma_rate': [0.9, 0.95, 1.0],
    }

    all_keys = list(model_search_space) + list(train_search_space)
    n_cols   = 4
    n_rows   = len(all_keys) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for idx, key in enumerate(all_keys):
        ax = axes[idx]
        if key in model_search_space:
            values = [run['best_model_cfg'].get(key) for run in entries]
            categories = model_search_space[key]
        else:
            values = [run['best_train_cfg'].get(key) for run in entries]
            categories = train_search_space[key]

        sns.countplot(x=values, order=categories, ax=ax)
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")

    # Hide any extra subplots
    for ax in axes[len(all_keys):]:
        ax.set_visible(False)

    fig.suptitle(f"Hyperparameter Distributions for {model_type}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = f"output/plots/hyperparam_distribution_{model_type}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Saved plot to {out_path}")