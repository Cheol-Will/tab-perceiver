import argparse
import math
import os
import os.path as osp
import time
from typing import Any, Optional

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.nn import LayerNorm, Linear, Sequential, Parameter
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Accuracy, Metric
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
from models import TabPerceiver

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
loss_fun = CrossEntropyLoss()
torch.manual_seed(args.seed)


def build_loader(args, dataset_index):
    print(f"Start building dataloder for data_{args.scale}_{dataset_index}")
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data')
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

    print('Training set has {} instances'.format(len(train_tensor_frame)))
    print('Validation set has {} instances'.format(len(val_tensor_frame)))
    print('Test set has {} instances'.format(len(test_tensor_frame)))

    return train_loader, valid_loader, test_loader, col_stats, col_names_dict, num_classes

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
        pred = model(tf)

        if pred.size(1) == 1:
            pred = pred.view(-1, )
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
    num_layers = 6
    num_latents = 32
    hidden_dim = 256
    dropout_prob = 0.2

    result_dicts = {}
    # For multiclass_classification, there exist three datsets.
    for i in range(0, 3):
        train_loader, valid_loader, test_loader, col_stats, col_names_dict, num_classes = build_loader(args, i)
        num_features = 0
        metric_computer = Accuracy(task='multiclass',
                                    num_classes=num_classes).to(device)
        for k, v in col_names_dict.items():
            num_features += len(v)    

        if i == 0:
            model = TabPerceiver(
                out_channels=num_classes,
                num_features=num_features,
                num_heads=num_heads,
                num_layers=num_layers,
                num_latents=num_latents,
                hidden_dim=hidden_dim,
                dropout_prob=dropout_prob,
                col_stats=col_stats,
                col_names_dict=col_names_dict
            ).to(device)
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
            # positional encoding for embedding
            model.pos_embedding = Parameter(torch.empty(1, num_features, hidden_dim))

            # make new decoder query
            model.queries = Parameter(torch.empty(1, 1, hidden_dim))
            
            # make new prediction head
            model.proj = Sequential(
                LayerNorm(hidden_dim),
                Linear(hidden_dim, num_classes)
            )           

            # initialize pos_embedding and queries
            model.reset_parameters_finetune()

        # check if new layers are requires_grad = True
        print(f"training {i}th dataset ")
        for name, params in model.named_parameters():
            if params.requires_grad:
                print(name)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        
        best_val_metric = math.inf
        best_test_metric = math.inf
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, epoch)
            val_metric = test(model, valid_loader, metric_computer)

            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_test_metric = test(model, test_loader, metric_computer)
            lr_scheduler.step()
            print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        print(f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')
        data_name = f"{args.scale}_{i}"
        result_dict = {
            "args": args.__dict__, # same for all dataset
            'best_val_metric': best_val_metric,
            'best_test_metric': best_test_metric,
        }
        result_dicts[data_name] = result_dict
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_dicts, args.result_path)

if __name__ == '__main__':
    main(args)