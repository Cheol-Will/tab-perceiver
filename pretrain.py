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
    torch.manual_seed(args.seed)
    result_list = []
    
    # basemodel    
    num_heads = 8
    num_layers = 4
    num_latents = 32
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

    pretrain_data_count = 0
    for task_type, dataset_indices_dict in dataset_index_ranges.items():
        for dataset_index in list(dataset_indices_dict["medium"]):
            data_name = f"{task_type}_medium_{dataset_index}"

            # build dataset
            if pretrain_data_count != 0:
                # do not re-build first dataset.
                print(f"{task_type}, dataset_scale, {dataset_index}")
                
                dataset = build_dataset(task_type=args.task_type, dataset_scale=args.scale, dataset_index=dataset_index)
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
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
            best_val_metric, best_test_metric = init_best_metric(higher_is_better) # regression: inf, classification: 0
                
            for epoch in range(args.epochs):
                train_loss = train(model, train_loader, optimizer, loss_fun, epoch)
                val_metric = test(model, valid_loader, metric_computer)

                if higher_is_better:
                    if val_metric > best_val_metric:
                        best_val_metric = val_metric
                        best_test_metric = test(model, test_loader)
                else:
                    if val_metric < best_val_metric:
                        best_val_metric = val_metric
                        best_test_metric = test(model, test_loader)
                lr_scheduler.step()
                
                print(f'Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}')

            result_list.append(
                {
                    data_name: {
                        "best_val_metric": best_val_metric,
                        "best_test_metric": best_test_metric
                    }
                }
            )
            print(f"Best val: {best_val_metric:.4f}, Best test: {best_test_metric:.4f}")
    chckpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if args.result_path != '':
        os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
        torch.save(result_list, args.result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='')
    args = parser.parse_args()
    main(args)