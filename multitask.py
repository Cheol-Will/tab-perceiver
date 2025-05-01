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
    for task_idx, dataloader in enumerate(dataloaders):
        task_order += [task_idx] * len(dataloader)
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
    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            pred = model(tf, task_idx)
            if task_type == TaskType.MULTICLASS_CLASSIFICATION:
                pred = pred.argmax(dim=-1)
            elif task_type == TaskType.REGRESSION:
                pred = pred.view(-1, )
            metric_computer.update(pred, tf.y)
    return metric_computer.compute().item()

def finetune_and_eval(
    model: Module,
    train_loaders,
    test_loaders,
    optimizer,
    lr_scheduler,
    loss_fun,
    finetune_epoch,
    metric_computer: Metric,
    task_type,
    task_idx,
) -> float:

    # create task_idx_list for one task
    finetune_task_idx_list = [task_idx] * len(train_loaders[task_idx])

    # train on finetune epoch
    for epoch in range(finetune_epoch):
        train_loss = train(model, train_loaders, optimizer, loss_fun, epoch, task_type, finetune_task_idx_list)
        lr_scheduler.step()
        print(f"Train Loss: {train_loss:.7f}")

    test_metric = test(model, test_loaders[task_idx], metric_computer, task_type, task_idx)
    return test_metric

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_config = {
        "num_heads": 8,
        "num_layers": 8,
        "num_latents": 16,
        "hidden_dim": 64,
        "dropout_prob": 0.2,
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
    result_list = [] 
    for task_idx in range(num_tasks):
        test_metric = test(model, test_loaders[task_idx], metric_computer, task_type, task_idx)
        result_list.append({
            f"{args.task_type}_{args.scale}_{task_idx}": {
                "best_test_metric": test_metric
            }
        })                
    # Save weight and result before fintune
    checkpoint = {
        "args": args.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "best_test_metric": result_list,
    }
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    torch.save(checkpoint, args.result_path) # Checkpoint before finetune
    print(f"Checkpoint before fintune is saved into {args.result_path}")

    # Finetune and eval for each task
    finetune_result_list = []
    for task_idx in range(num_tasks):
        # load state_dict to finetune each task from the same weight
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        finetune_test_metric = finetune_and_eval(
            model,
            train_loaders,
            test_loaders,
            optimizer,
            lr_scheduler,
            loss_fun,
            args.finetune_epochs,
            metric_computer,
            task_type,
            task_idx
        )

        finetune_result_list.append({
            f"{args.task_type}_{args.scale}_{task_idx}": {
                "finetune_best_test_metric": finetune_test_metric
            }
        })        

    checkpoint["finetune_test_metric"] = finetune_result_list
    torch.save(checkpoint, args.result_path)
    print(f"Checkpoint after finetune is saved into {args.result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['binary_classification', 'multiclass_classification', 'regression'],
                        default='binary_classification')
    parser.add_argument('--scale', type=str, choices=['small', 'medium', 'large'],
                        default='small')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--result_path', type=str, default='')
    args = parser.parse_args()
    main(args)