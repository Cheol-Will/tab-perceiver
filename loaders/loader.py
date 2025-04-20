import os.path as osp
from torch_frame.typing import TaskType
from torch_frame.data import DataLoader
from torch_frame.datasets import DataFrameBenchmark

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

    col_stats = dataset.col_stats
    col_names_dict = train_tensor_frame.col_names_dict
    num_features = 0
    for k, v in col_names_dict.items():
        num_features += len(v)   

    meta_data = {
        "col_stats": col_stats,
        "col_names_dict": col_names_dict,
        "num_features": num_features,
    }

    return train_loader, valid_loader, test_loader, meta_data