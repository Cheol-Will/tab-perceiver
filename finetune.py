import argparse
import math
import os
from typing import Any, Optional

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm

from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverTransfer
from utils import create_train_setup, init_best_metric
from loaders import build_dataset, build_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fun: Module,
    epoch: int,
    task_type,
) -> float:
    model.train()
    loss_accum = total_count = 0

    for tf in tqdm(loader, desc=f'Epoch: {epoch}'):
        tf = tf.to(device)
        y = tf.y
        pred = model(tf)
        if torch.isnan(pred).any():
            print("NAN in pred")
        if pred.size(1) == 1:
            pred = pred.view(-1, )
        if task_type == TaskType.BINARY_CLASSIFICATION:
            y = y.to(torch.float)
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
    task_type,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()


def main(args):
    """ train on first dataset and finetune on other datasets"""
    torch.manual_seed(args.seed)

    num_heads = 8
    num_layers = 8
    num_latents = 8
    hidden_dim = 32
    dropout_prob = 0.2

    dataset = build_dataset(args.task_type, dataset_scale="medium", dataset_index=0)
    train_loader, valid_loader, test_loader, meta_data = build_dataloader(dataset)
    out_channels, loss_fun, metric_computer, higher_is_better = create_train_setup(dataset)
    metric_computer.to(device)

    model = TabPerceiverTransfer(
        out_channels=out_channels, 
        num_features=meta_data["num_features"],
        num_heads=num_heads,
        num_layers=num_layers,
        num_latents=num_latents,
        hidden_dim=hidden_dim,
        dropout_prob=dropout_prob,
        col_stats=meta_data["col_stats"],
        col_names_dict=meta_data["col_names_dict"]
    ).to(device)

    result_dicts = {}
    for i in range(3):

        if i != 0:
            dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=i)
            train_loader, valid_loader, test_loader, meta_data = build_dataloader(dataset)
            out_channels, loss_fun, metric_computer, higher_is_better, task_type = create_train_setup(dataset)
            metric_computer.to(device)

            # reconstruct input and output layers
            model.reconstructIO(
                out_channels=out_channels,
                num_features=meta_data["num_features"],
                col_stats=meta_data["col_stats"],
                col_names_dict=meta_data["col_names_dict"],
            )
            model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
        best_val_metric, best_test_metric = init_best_metric(higher_is_better) # regression: inf, classification: 0
            
        for epoch in range(1, args.epochs + 1):
            train_loss = train(model, train_loader, optimizer, loss_fun, epoch, task_type)
            val_metric = test(model, valid_loader, metric_computer, task_type)

            if higher_is_better:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test(model, test_loader, metric_computer, task_type)
            else:
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    best_test_metric = test(model, test_loader, metric_computer, task_type)
            lr_scheduler.step()
            print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

        print(f'Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}')
        data_name = f"{args.task_type}_{args.scale}_{i}"

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
    main(args)