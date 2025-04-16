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
from torch.nn import LayerNorm, Linear, Sequential, Parameter
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import AUROC, Accuracy, MeanSquaredError
from tqdm import tqdm

from torch_frame import stype
from torch_frame import TensorFrame, stype
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark
from torch_frame.nn.encoder.stype_encoder import (
    EmbeddingEncoder,
    LinearEncoder,
)
from torch_frame.nn.encoder.stypewise_encoder import StypeWiseFeatureEncoder
from models import TabPerceiver

from data_frame_benchmark import train, test

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
        'TabPerceiver'
    ])
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--result_path', type=str, default='')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

def build_loader(args, dataset_index):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
    # For multiclass_classification, there exist three datsets.
    dataset = DataFrameBenchmark(root=path, task_type=TaskType(args.task_type),
                                scale=args.scale, idx=dataset_index)
    dataset.materialize()
    dataset = dataset.shuffle()
    train_dataset, val_dataset, test_dataset = dataset.split()

    train_tensor_frame = train_dataset.tensor_frame
    val_tensor_frame = val_dataset.tensor_frame
    test_tensor_frame = test_dataset.tensor_frame

    col_stats = dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict
    num_classes = dataset.num_classes

    train_loader = DataLoader(train_tensor_frame, batch_size=128, shuffle=True, drop_last=True)
    valid_loader = DataLoader(val_tensor_frame, batch_size=128)
    test_loader = DataLoader(test_tensor_frame, batch_size=128)

    return train_loader, valid_loader, test_loader, col_stats, col_names_dict, num_classes

def main(args):
    num_heads = 8
    hidden_dim = 256
    num_layers = 6
    num_latents = 32
    hidden_dim = 256
    dropout_prob = 0.2

    for i in range(0, 3):
        train_loader, valid_loader, test_loader, col_stats, col_names_dict, num_classes = build_loader(args, i)
        if i == 0:
            model = TabPerceiver(
                num_classes,
                num_heads,
                num_layers,
                num_latents,
                hidden_dim,
                dropout_prob,
                col_stats=col_stats,
                col_names_dict=col_names_dict
            )
        else:
            # freeze
            for param in model.parameters():
                param.requires_grad = False

            # make new feature embedding layer            
            stype_encoder_dict = {
                stype.categorical: EmbeddingEncoder(),
                stype.numerical: LinearEncoder(),
            }
            model.tensor_frame_encoder = StypeWiseFeatureEncoder(
                out_channels=hidden_dim,
                col_stats=col_stats,
                col_names_dict=col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
            )
            # make new decoder query
            model.queries = Parameter(torch.empty(1, 1, hidden_dim))
            torch.nn.init.trunc_normal_(model.queries, std=0.02)
            
            # make new prediction head
            model.proj = Sequential(
                LayerNorm(hidden_dim),
                Linear(hidden_dim, num_classes)
            )           

        # check if new layers are requires_grad = True
        # print(f"training {i}th dataset ")
        # for name, params in model.named_parameters():
        #     if params.requires_grad:
        #         print(name)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        best_val_metric = math.inf
        best_test_metric = math.inf
        
        for epoch in range(1, args.epoch + 1):
            train_loss = train(model, train_loader, optimizer, epoch)
            val_metric = test(model, valid_loader)

            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader)
            lr_scheduler.step()
            print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        print(f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')


if __name__ == '__main__':
    main(args)