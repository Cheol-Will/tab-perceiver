import argparse
import math
import os
import os.path as osp
import time
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Module
from torch.nn import LayerNorm, Linear, Sequential, Parameter
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Accuracy, AUROC, MeanSquaredError, Metric
from tqdm import tqdm

from torch_frame import TensorFrame, stype
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from models import TabPerceiverTransfer

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task_type', type=str, choices=[
        'binary_classification',
        'multiclass_classification',
        'regression',
    ], default='binary_classification')
parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                    default='small')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--result_path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

dataset_index_ranges = {
    'binary_classification': {
        'small':   range(0, 14),  'medium': range(0, 9),  'large': range(0, 1),
    },
    'multiclass_classification': {
        'small':   [],            'medium': range(0, 3),  'large': range(0, 3),
    },
    'regression': {
        'small':   range(0, 13),  'medium': range(0, 6),  'large': range(0, 6),
    },
}


def build_dataset(task_type, dataset_scale, dataset_index):
    print(f"Start building {task_type} dataset_{dataset_scale}_{dataset_index}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    dataset = DataFrameBenchmark(root=path, task_type=TaskType(task_type),
                                scale=dataset_scale, idx=dataset_index)
    dataset.materialize()
    return dataset

def build_dataloader(dataset):
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame

    train_loader = DataLoader(train_tensor_frame, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_tensor_frame, batch_size=128)
    test_loader = DataLoader(test_tensor_frame, batch_size=128)

    print(f'Training set has {len(train_tensor_frame)} instances')
    print(f'Validation set has {len(val_tensor_frame)} instances')
    print(f'Test set has {len(test_tensor_frame)} instances')
    print(f"Number of classes: {dataset.num_classes}")

    col_stats = dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict

    return train_loader, valid_loader, test_loader, col_stats, col_names_dict


def create_train_setup(dataset):
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary').to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes).to(device)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False).to(device)
        higher_is_better = False

    return out_channels, loss_fun, metric_computer, higher_is_better 


def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fun: Module,
    epoch: int,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        pred = model(tf)
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
    metric_computer: Metric,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        pred = pred.argmax(dim=-1)
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def main(args):
    num_heads = 8
    num_layers = 4
    num_latents = 32
    hidden_dim = 32
    dropout_prob = 0.2

    result_dicts = {}
    
    # basemodel    
    num_heads = 8
    num_layers = 4
    num_latents = 32
    hidden_dim = 32
    dropout_prob = 0.2

    dataset = build_dataset(binary_classification, dataset_scale="medium", dataset_index=0)
    train_loader, valid_loader, test_loader, col_stats, col_names_dict = build_dataloader(args, dataset)
    out_channels, loss_fun, metric_computer, higher_is_better = create_train_setup(dataset)
    num_features = 0
        for k, v in col_names_dict.items():
            num_features += len(v)    
    
    model = TabPerceiverTransfer(
                    out_channels=1,
                    num_features=num_features,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    num_latents=num_latents,
                    hidden_dim=hidden_dim,
                    dropout_prob=dropout_prob,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict
    ).to(device)

    pretrain_data_count = 0
    for task_type, dataset_indices_dict in dataset_index_ranges.items():
        for dataset_index in list(dataset_indices_dict["medium"]):

            # build datast
            if pretrain_data_count != 0:
                # do not re-build first dataset.
                print(f"{task_type}, dataset_scale, {dataset_index}")
                dataset = build_dataset(task_type, dataset_scale="medium", dataset_index):
                train_loader, valid_loader, test_loader, col_stats, col_names_dict = build_dataloader(args, dataset)
                out_channels, loss_fun, metric_computer, higher_is_better = create_train_setup(dataset)

            num_features = 0
            for k, v in col_names_dict.items():
                num_features += len(v)    

            # reconstruct input and output layers
            model.reconstructIO(
                out_channels: out_channels,
                num_features: num_features,
                col_stats: col_stats,
                col_names_dict: col_names_dict,
            )

            


if __name__ == '__main__':
    main(args)