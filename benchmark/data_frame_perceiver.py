import argparse
import math
import os
import os.path as osp
import time
from typing import Any, Optional

import numpy as np
import optuna
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, Module, MSELoss
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame import stype
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.gbdt import CatBoost, LightGBM, XGBoost
from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.nn.models import (
    MLP,
    ExcelFormer,
    FTTransformer,
    ResNet,
    TabNet,
    TabTransformer,
    Trompt,
    FTPerceiver,
)
from torch_frame.typing import TaskType

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
GBDT_MODELS = ["XGBoost", "CatBoost", "LightGBM"]

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--idx', type=int, default=0,
                    help='The index of the dataset within DataFrameBenchmark')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--num_trials', type=int, default=20,
                    help='Number of Optuna-based hyper-parameter tuning.')
parser.add_argument(
    '--num_repeats', type=int, default=5,
    help='Number of repeated training and eval on the best config.')
parser.add_argument(
    '--model_type', type=str, default='TabNet', choices=[
        'TabNet', 'FTTransformer', 'ResNet', 'MLP', 'TabTransformer', 'Trompt',
        'ExcelFormer', 'FTTransformerBucket', 'XGBoost', 'CatBoost', 'LightGBM',
        'FTPerceiver'
    ])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--result_path', type=str, default='output/result.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

# Prepare datasets
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                             scale=args.scale, idx=args.idx)
dataset.materialize()
dataset = dataset.shuffle()
train_dataset, val_dataset, test_dataset = dataset.split()

train_tensor_frame = train_dataset.tensor_frame
val_tensor_frame = val_dataset.tensor_frame
test_tensor_frame = test_dataset.tensor_frame

out_channels = dataset.num_classes
col_stats = dataset.col_stats
col_names_dict = train_tensor_frame.col_names_dict

model = FTTransformer(
    channels=64,
    out_channels=out_channels,
    num_layers=4,
    col_stats=col_stats,
    col_names_dict=col_names_dict,
)
# model = FTPerceiver(channels=1, 
#                     out_channels=out_channels, 
#                     num_heads=8, 
#                     num_layers=4, 
#                     num_latent_array=8, 
#                     latent_channels=64, 
#                     col_stats=col_stats,
#                     col_names_dict=col_names_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fun = BCEWithLogitsLoss()

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        if isinstance(model, ExcelFormer):
            # Train with FEAT-MIX or HIDDEN-MIX
            pred, y = model(tf, mixup_encoded=True)
        elif isinstance(model, Trompt):
            # Trompt uses the layer-wise loss
            pred = model(tf)
            num_layers = pred.size(1)
            # [batch_size * num_layers, num_classes]
            pred = pred.view(-1, out_channels)
            y = tf.y.repeat_interleave(num_layers)
        else:
            pred = model(tf)

        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)

        print(f"shape of pred: {pred.shape}")
        print(f"shape of y: {y.shape}")
        loss = loss_fun(pred, y)
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
    return loss_accum / total_count


def test(
    model: Module,
    loader: DataLoader,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if isinstance(model, Trompt):
            pred = pred.mean(dim=1)
        if dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif dataset.task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()

train(model, train_tensor_frame, optimizer, 1)