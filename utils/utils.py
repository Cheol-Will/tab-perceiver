import math
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Module
from torchmetrics import Accuracy, AUROC, MeanSquaredError, Metric
from torch_frame.typing import TaskType

def create_train_setup(dataset):
    if dataset.task_type == TaskType.BINARY_CLASSIFICATION:
        out_channels = 1
        loss_fun = BCEWithLogitsLoss()
        metric_computer = AUROC(task='binary')
        higher_is_better = True
    elif dataset.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        out_channels = dataset.num_classes
        loss_fun = CrossEntropyLoss()
        metric_computer = Accuracy(task='multiclass',
                                   num_classes=dataset.num_classes)
        higher_is_better = True
    elif dataset.task_type == TaskType.REGRESSION:
        out_channels = 1
        loss_fun = MSELoss()
        metric_computer = MeanSquaredError(squared=False)
        higher_is_better = False

    return out_channels, loss_fun, metric_computer, higher_is_better, dataset.task_type


def init_best_metric(higher_is_better):
    if higher_is_better:
        best_val_metric = 0
        best_test_metric = 0
    else:
        best_val_metric = math.inf
        best_test_metric = math.inf

    return best_val_metric, best_test_metric