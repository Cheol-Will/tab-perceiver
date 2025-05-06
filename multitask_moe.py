import argparse
import os
import random

import torch
from torch.nn import Module
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics import Metric
from tqdm import tqdm
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from models import TabPerceiverMultiTask
from utils import create_multitask_setup, init_best_metric
from loaders import build_datasets, build_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def shuffle_task_index(dataloaders):
    task_order = []
    for taks_index, dataloader in enumerate(dataloaders):
        task_order += [taks_index] * len(dataloader)
    random.shuffle(task_order)
    return task_order

def train(
    model: Module,
    loaders,
    optimizer: torch.optim.Optimizer,
    loss_fun: Module,
    epoch: int,
    task_type,
    task_idx_list
) -> float:
    model.train()
    loss_accum = total_count = 0

    loaders = [iter(loader) for loader in loaders]

    for task_idx in tqdm(task_idx_list, desc=f'Epoch: {epoch}'):
        tf = next(loaders[task_idx])    
        tf = tf.to(device)
        y = tf.y
        pred = model(tf, task_idx)
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
    task_idx,
) -> float:
    model.eval()
    metric_computer.reset()
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf, task_idx)
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            pred = pred.argmax(dim=-1)
        elif task_type == TaskType.REGRESSION:
            pred = pred.view(-1, )
        metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()

def main(args):
    torch.manual_seed(args.seed)
    model_config = {
        "num_heads": 4,
        "num_layers": 6,
        "num_latents": 16,
        "hidden_dim": 64,
        "mlp_ratio": 2,
        "moe_ratio": 0.25,
        "dropout_prob": 0,
        "is_moe": True,
    }

    datasets = build_datasets(task_type=args.task_type, dataset_scale=args.scale)
    num_tasks = len(datasets)
    train_loaders, valid_loaders, test_loaders, meta_data = build_dataloaders(datasets)
    num_classes, loss_fun, metric_computer, higher_is_better, task_type = create_multitask_setup(datasets)
    metric_computer.to(device)

    model = TabPerceiverMultiTask(
        **model_config,
        **meta_data,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(args.epochs):        
        task_idx_list = shuffle_task_index(train_loaders) # sample task_idx list
        train_loss = train(model, train_loaders, optimizer, loss_fun, epoch, task_type, task_idx_list)
        lr_scheduler.step()
        print(f"Train Loss: {train_loss:.7f}")

    # test on every task
    for task_idx in range(num_tasks):
        test_metric = test(model, test_loaders[task_idx], metric_computer, task_type, task_idx)
        print(f"[Task {task_idx}]")
        print(f"Test metric: {test_metric:.6f}")
        result_dict = {
            'args': args.__dict__,
            'model_config': model_config,
            "best_test_metric": test_metric
        }
        path = f"output/{args.task_type}/{args.scale}/{task_idx}/{args.exp_name}.pt"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(result_dict, path)
        print(f"Result is saved into {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='TabPerceiverMultiTaskMOE')
    args = parser.parse_args()
    main(args)